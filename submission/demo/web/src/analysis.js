// Only the versioned offline artifact selects/pairs runs or makes inferential decisions.
// Missing provenance never falls back to matching seed numbers in the browser.
export const CONTRAST_ARMS = {
  "g8-ppo": ["grpo_g8", "ppo"],
  "g8-ppo_8x8": ["grpo_g8", "ppo_8x8"],
  "g8-g2": ["grpo_g8", "grpo_g2"],
  "g2-ppo": ["grpo_g2", "ppo"],
};
export const PRIMARY_IDS = ["g8-ppo", "g8-ppo_8x8", "g8-g2"];
const finite = (v) => typeof v === "number" && Number.isFinite(v);
const probability = (v) => finite(v) && v >= 0 && v <= 1;
const identity = ["run_id", "arm", "seed", "environment", "environment_hash", "shared_stack_hash", "protocol_hash", "ordered_test_item_hash"];

export function analysisModel(data) {
  const raw = data.analysis;
  const analysis = raw?.schema_version === 1 ? raw : null;
  const runs = Array.isArray(data.samestack?.runs) ? data.samestack.runs : [];
  const selections = Array.isArray(analysis?.selected_runs) ? analysis.selected_runs : [];
  const selectedRuns = selections.flatMap((s) => {
    if (!s || s.completion_bundle_validated !== true) return [];
    if (!identity.every((k) => k === "seed" ? Number.isInteger(s[k]) : typeof s[k] === "string" && s[k].length > 0)) return [];
    const matches = runs.filter((r) => r?.completion_bundle_validated === true && identity.every((k) => r[k] === s[k]));
    return matches.length === 1 ? matches : [];
  });
  const selectionValid = selectedRuns.length === selections.length &&
    new Set(selections.map((r) => r?.run_id)).size === selections.length &&
    (!analysis || analysis.completion?.selected === selections.length);
  const plan = analysis?.plan || {};
  const bindingsValid = ["environment", "environment_hash", "shared_stack_hash", "protocol_hash", "ordered_test_item_hash"]
    .every((k) => typeof plan[k] === "string" && plan[k].length > 0 && selections.every((r) => r?.[k] === plan[k]));
  const familyValid = Array.isArray(plan.primary_contrast_ids) &&
    plan.primary_contrast_ids.length === PRIMARY_IDS.length && PRIMARY_IDS.every((id) => plan.primary_contrast_ids.includes(id));
  const alpha = probability(plan.alpha) && plan.alpha > 0 ? plan.alpha : 0.05;
  const margin = finite(plan.equivalence_margin) && plan.equivalence_margin > 0 ? plan.equivalence_margin : null;
  const justified = typeof plan.margin_justification === "string" && plan.margin_justification.trim().length > 0;
  const endpointComplete = analysis?.status === "endpoint_complete" && plan.endpoint?.declared === true &&
    analysis?.completion?.missing === 0 && selectionValid && bindingsValid && familyValid;
  const complete = endpointComplete && analysis?.final_eligible === true;
  const contrasts = [...PRIMARY_IDS, "g2-ppo"].map((id) => {
    const exploratory = !PRIMARY_IDS.includes(id);
    const c = (exploratory ? analysis?.exploratory_contrasts : analysis?.primary_contrasts)?.[id];
    const [a, b] = CONTRAST_ARMS[id];
    const pairs = Array.isArray(c?.pairs) ? c.pairs : [];
    const pairValid = pairs.length === c?.n && pairs.every((pair, index) => {
      if (!pair || pair.seed !== c.seeds?.[index]) return false;
      const x = selectedRuns.find((r) => r.run_id === pair.a_run_id);
      const y = selectedRuns.find((r) => r.run_id === pair.b_run_id);
      return x && y && x.arm === a && y.arm === b && x.seed === pair.seed && y.seed === pair.seed &&
        ["environment", "environment_hash", "shared_stack_hash", "protocol_hash", "ordered_test_item_hash"].every((k) => x[k] === y[k]);
    });
    const valid = c && Number.isInteger(c.n) && c.n >= 2 && pairValid &&
      Array.isArray(c.seeds) && c.seeds.length === c.n && c.seeds.every(Number.isInteger) && new Set(c.seeds).size === c.n &&
      Array.isArray(c.diffs) && c.diffs.length === c.n && c.diffs.every(finite) && finite(c.mean) &&
      Array.isArray(c.t_ci) && c.t_ci.length === 2 && c.t_ci.every(finite) && c.t_ci[0] <= c.mean && c.mean <= c.t_ci[1] &&
      Array.isArray(c.tost_ci90) && c.tost_ci90.length === 2 && c.tost_ci90.every(finite) && c.tost_ci90[0] <= c.mean && c.mean <= c.tost_ci90[1] &&
      [c.p_t, c.tost_p].every(probability) &&
      (c.p_signflip === null || probability(c.p_signflip)) &&
      (exploratory || ([c.holm_p_t, c.holm_tost_p].every(probability) &&
        (c.holm_p_signflip === null || probability(c.holm_p_signflip)))) && selectionValid && bindingsValid;
    const stats = valid ? {
      n: c.n, md: c.mean, lo: c.t_ci[0], hi: c.t_ci[1],
      eqLo: c.tost_ci90[0], eqHi: c.tost_ci90[1], p: c.p_t, sf: c.p_signflip,
      tost: c.tost_p, holmP: c.holm_p_t, holmSF: c.holm_p_signflip, holmTost: c.holm_tost_p,
      positive5pp: c.positive_5pp_test?.p_value ?? null,
    } : null;
    const equivalent = Boolean(stats && complete && c.equivalence_claim_eligible === true && !exploratory && justified && margin &&
      probability(stats.holmTost) && stats.holmTost < alpha);
    return { id, a, b, exploratory, stats, seeds: c?.seeds || [], diffs: c?.diffs || [],
      missingSeeds: c?.missing_seeds || [], equivalent,
      decision: equivalent ? `Equivalence established within ±${(100 * margin).toFixed(1)} percentage points` :
        exploratory ? "Exploratory comparison; no confirmatory verdict" :
        !complete ? "Pending: no confirmatory verdict without a validated final analysis" : "Equivalence not established" };
  });
  return { analysis, selectedRuns, selectionValid, complete, endpointComplete, plan, alpha, margin, justified, contrasts,
    reasons: analysis?.interim_reasons || ["Canonical analysis artifact unavailable; historical records are descriptive only"],
    completion: analysis?.completion || { planned: null, selected: 0, missing: null } };
}

export function armRows(runs) {
  const groups = {};
  for (const r of runs) {
    if (typeof r.arm === "string" && finite(r.pre) && finite(r.post)) (groups[r.arm] ||= []).push(r);
  }
  return Object.entries(groups).map(([arm, rs]) => ({ arm, n: rs.length,
    pre: rs.reduce((n, r) => n + r.pre, 0) / rs.length,
    posts: rs.map((r) => r.post),
    post: rs.reduce((n, r) => n + r.post, 0) / rs.length,
    gain: rs.reduce((n, r) => n + r.post - r.pre, 0) / rs.length,
  }));
}
