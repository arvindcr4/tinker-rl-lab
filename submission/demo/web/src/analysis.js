import { paired, holm } from "./stats.js";
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
  const plan = analysis?.plan || {};
  const plannedSeeds = plan.planned_seeds;
  const cohortValid = typeof plan.plan_id === "string" && plan.plan_id.length > 0 &&
    Array.isArray(plannedSeeds) && plannedSeeds.length >= 2 && plannedSeeds.every(Number.isInteger) &&
    new Set(plannedSeeds).size === plannedSeeds.length &&
    selectedRuns.every((r) => ["grpo_g8", "grpo_g2", "ppo", "ppo_8x8"].includes(r.arm) &&
      plannedSeeds.includes(r.seed) && r.manifest?.analysis_plan_id === plan.plan_id) &&
    new Set(selectedRuns.map((r) => `${r.arm}:${r.seed}`)).size === selectedRuns.length;
  const selectionValid = cohortValid && selectedRuns.length === selections.length &&
    new Set(selections.map((r) => r?.run_id)).size === selections.length &&
    (!analysis || analysis.completion?.selected === selections.length);
  const bindingsValid = ["environment", "environment_hash", "shared_stack_hash", "protocol_hash", "ordered_test_item_hash"]
    .every((k) => typeof plan[k] === "string" && plan[k].length > 0 && selections.every((r) => r?.[k] === plan[k]));
  const familyValid = Array.isArray(plan.primary_contrast_ids) &&
    plan.primary_contrast_ids.length === PRIMARY_IDS.length && PRIMARY_IDS.every((id) => plan.primary_contrast_ids.includes(id));
  const alpha = probability(plan.alpha) && plan.alpha > 0 ? plan.alpha : 0.05;
  const margin = finite(plan.equivalence_margin) && plan.equivalence_margin > 0 ? plan.equivalence_margin : null;
  const justified = typeof plan.margin_justification === "string" && plan.margin_justification.trim().length > 0;
  const endpointComplete = analysis?.status === "endpoint_complete" && plan.endpoint?.declared === true &&
    analysis?.completion?.missing === 0 && analysis?.completion?.planned === plannedSeeds?.length * 4 &&
    selections.length === plannedSeeds?.length * 4 && selectionValid && bindingsValid && familyValid;
  const complete = endpointComplete && analysis?.final_eligible === true;
  // Recompute only to verify the canonical artifact, never to fill in absent inference.
  const close = (x, y) => x === null || y === null ? x === y :
    finite(x) && finite(y) && Math.abs(x - y) <= 1e-9 * Math.max(1, Math.abs(x), Math.abs(y));
  const verified = {};
  for (const [id, [a, b]] of Object.entries(CONTRAST_ARMS)) {
    const c = (PRIMARY_IDS.includes(id) ? analysis?.primary_contrasts : analysis?.exploratory_contrasts)?.[id];
    if (!selectionValid || !bindingsValid || !margin || !c || !Number.isInteger(c.n) || c.n < 2 ||
        !Array.isArray(c.pairs) || c.pairs.length !== c.n || !Array.isArray(c.seeds) || c.seeds.length !== c.n ||
        !c.seeds.every(Number.isInteger) || new Set(c.seeds).size !== c.n ||
        !Array.isArray(c.diffs) || c.diffs.length !== c.n) continue;
    const expectedSeeds = plannedSeeds.filter((seed) => selectedRuns.some((r) => r.arm === a && r.seed === seed) &&
      selectedRuns.some((r) => r.arm === b && r.seed === seed));
    if (expectedSeeds.length !== c.n || !expectedSeeds.every((seed, i) => seed === c.seeds[i])) continue;
    const diffs = [];
    const pairsValid = c.pairs.every((pair, i) => {
      if (!pair || pair.seed !== c.seeds[i]) return false;
      const x = selectedRuns.find((r) => r.run_id === pair.a_run_id);
      const y = selectedRuns.find((r) => r.run_id === pair.b_run_id);
      if (!x || !y || x.arm !== a || y.arm !== b || x.seed !== pair.seed || y.seed !== pair.seed ||
          !probability(x.post) || !probability(y.post)) return false;
      diffs.push(x.post - y.post);
      return close(c.diffs[i], diffs[i]);
    });
    if (!pairsValid) continue;
    const expected = paired(diffs, margin);
    if (!close(c.mean, expected.md) || !close(c.p_t, expected.p) || !close(c.tost_p, expected.tost) ||
        !close(c.p_signflip, expected.sf) ||
        !Array.isArray(c.t_ci) || c.t_ci.length !== 2 || !c.t_ci.every((v, i) => close(v, expected.difference_ci_95[i])) ||
        !Array.isArray(c.tost_ci90) || c.tost_ci90.length !== 2 || !c.tost_ci90.every((v, i) => close(v, expected.equivalence_ci_90[i]))) continue;
    verified[id] = expected;
  }
  const adjusted = Object.fromEntries(["p", "sf", "tost"].map((field) => [field,
    holm(Object.fromEntries(PRIMARY_IDS.map((id) => [id, verified[id]?.[field] ?? 1])))]));
  const contrasts = [...PRIMARY_IDS, "g2-ppo"].map((id) => {
    const exploratory = !PRIMARY_IDS.includes(id);
    const c = (exploratory ? analysis?.exploratory_contrasts : analysis?.primary_contrasts)?.[id];
    const [a, b] = CONTRAST_ARMS[id];
    const valid = verified[id] && (exploratory ||
      (close(c.holm_p_t, adjusted.p[id]) && close(c.holm_tost_p, adjusted.tost[id]) &&
       close(c.holm_p_signflip, adjusted.sf[id])));
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
