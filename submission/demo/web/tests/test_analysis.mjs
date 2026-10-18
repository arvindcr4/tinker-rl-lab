import assert from 'node:assert/strict';
import test from 'node:test';
import { analysisModel, armRows, PRIMARY_IDS } from '../src/analysis.js';

function fixture() {
  const runs = ['grpo_g8', 'ppo', 'ppo_8x8', 'grpo_g2'].flatMap((arm) => [1, 2].map((seed) => ({
    run_id: `${arm}-${seed}`, arm, seed, completion_bundle_validated: true, environment: 'fixture', environment_hash: 'hardware-and-dependencies', shared_stack_hash: 'effective-stack', protocol_hash: 'protocol',
    ordered_test_item_hash: 'items', manifest: { analysis_plan_id: 'fixture-plan' }, pre: .4, post: .5,
  })));
  const arms = { 'g8-ppo': ['grpo_g8', 'ppo'], 'g8-ppo_8x8': ['grpo_g8', 'ppo_8x8'], 'g8-g2': ['grpo_g8', 'grpo_g2'], 'g2-ppo': ['grpo_g2', 'ppo'] };
  const contrast = (id) => ({ n: 2, seeds: [1, 2], diffs: [0, 0], mean: 0,
    pairs: [1, 2].map((seed) => ({ seed, a_run_id: `${arms[id][0]}-${seed}`, b_run_id: `${arms[id][1]}-${seed}` })),
    t_ci: [0, 0], tost_ci90: [0, 0], p_t: 1, p_signflip: 1,
    tost_p: 0, holm_p_t: 1, holm_p_signflip: 1, holm_tost_p: 0, equivalence_claim_eligible: true,
  });
  return { samestack: { runs }, analysis: { schema_version: 1, status: 'endpoint_complete', final_eligible: true,
    plan: { plan_id: 'fixture-plan', planned_seeds: [1, 2], environment: 'fixture', environment_hash: 'hardware-and-dependencies', shared_stack_hash: 'effective-stack', protocol_hash: 'protocol', ordered_test_item_hash: 'items', primary_contrast_ids: PRIMARY_IDS, endpoint: { declared: true }, equivalence_margin: .02,
      margin_justification: 'Explicit test fixture only', alpha: .05 },
    completion: { planned: 8, selected: 8, missing: 0 }, selected_runs: runs.map(({ pre, post, ...r }) => r),
    primary_contrasts: Object.fromEntries(PRIMARY_IDS.map((id) => [id, contrast(id)])),
    exploratory_contrasts: { 'g2-ppo': contrast('g2-ppo') }, interim_reasons: [],
  }};
}

test('no canonical artifact, empty arms, and historical seed overlap never create paired inference', () => {
  for (const runs of [[], [{ arm: 'grpo_g8', seed: 42, pre: .6, post: .7 }, { arm: 'ppo', seed: 42, pre: .5, post: .6 }]]) {
    const m = analysisModel({ samestack: { runs } });
    assert.equal(m.complete, false);
    assert.equal(m.contrasts.length, 4);
    assert.ok(m.contrasts.every((c) => !c.stats && !c.equivalent));
  }
});
test('verified canonical artifact drives intervals and adjusted decision; exploratory excluded', () => {
  const m = analysisModel(fixture());
  assert.equal(m.contrasts[0].stats.eqLo, 0);
  assert.equal(m.contrasts[0].stats.holmTost, 0);
  assert.equal(m.contrasts[0].equivalent, true);
  assert.equal(m.contrasts[3].equivalent, false);
});
test('interim, retrospective, unjustified margin, absent arm, and unadjusted-only significance cannot establish equivalence', () => {
  const changes = [
    (d) => { d.analysis.status = 'interim'; },
    (d) => { d.analysis.final_eligible = false; },
    (d) => { d.analysis.plan.margin_justification = null; },
    (d) => { d.analysis.plan.endpoint.declared = false; },
    (d) => { d.analysis.plan.environment_hash = 'different-GPU'; },
    (d) => { d.analysis.plan.shared_stack_hash = null; },
    (d) => { d.analysis.completion.missing = 1; },
    (d) => { d.analysis.primary_contrasts['g8-ppo'].holm_tost_p = .06; },
    (d) => { delete d.analysis.primary_contrasts['g8-ppo'].holm_tost_p; },
    (d) => { d.analysis.plan.primary_contrast_ids = ['g8-ppo']; },
    (d) => { d.samestack.runs = d.samestack.runs.filter((r) => r.arm !== 'ppo'); },
  ];
  for (const change of changes) {
    const d = fixture(); change(d);
    assert.equal(analysisModel(d).contrasts[0].equivalent, false);
  }
});
test('zero contrasts, one pair, disjoint or incompatible identities remain unavailable', () => {
  const changes = [
    (d) => { d.analysis.primary_contrasts = {}; d.analysis.exploratory_contrasts = {}; },
    (d) => { d.analysis.primary_contrasts['g8-ppo'].n = 1; },
    (d) => { d.analysis.primary_contrasts['g8-ppo'].pairs[0].a_run_id = 'missing'; },
    (d) => { d.samestack.runs[0].completion_bundle_validated = false; },
    (d) => { d.analysis.selected_runs[0].completion_bundle_validated = false; },
    (d) => { d.samestack.runs[0].environment = 'other'; },
    (d) => { d.samestack.runs[0].environment_hash = 'different-GPU'; },
    (d) => { d.samestack.runs[0].shared_stack_hash = 'different-source-or-revision'; },
    (d) => { d.samestack.runs[0].protocol_hash = 'other'; },
    (d) => { d.samestack.runs[0].ordered_test_item_hash = 'other'; },
    (d) => { d.samestack.runs.push({ ...d.samestack.runs[0] }); },
  ];
  for (const change of changes) { const d = fixture(); change(d); assert.equal(analysisModel(d).contrasts[0].stats, null); }
});
test('descriptive rows use each arm own pre-score and retain duplicate seed records', () => {
  const rows = armRows([{ arm: 'a', seed: 1, pre: .2, post: .4 }, { arm: 'a', seed: 1, pre: .4, post: .4 }, { arm: 'b', seed: 1, pre: .8, post: .6 }]);
  assert.equal(rows[0].n, 2);
  assert.ok(Math.abs(rows[0].pre - .3) < 1e-12);
  assert.ok(Math.abs(rows[1].gain + .2) < 1e-12);
  assert.deepEqual(armRows([]), []);
});

test('malformed or misaligned inference fields cannot produce a confirmatory badge', () => {
  const changes = [
    (c) => { c.seeds = [2, 1]; },
    (c) => { c.seeds = '12'; },
    (c) => { c.diffs = { length: 2 }; },
    (c) => { c.pairs = {}; },
    (c) => { c.pairs[0] = null; },
    (c) => { c.t_ci = [.1, -.1]; },
    (c) => { c.tost_ci90 = [1, 2]; },
    (c) => { c.tost_p = NaN; },
    (c) => { c.p_t = -1; },
    (c) => { c.holm_p_t = 2; },
  ];
  for (const change of changes) {
    const d = fixture(); change(d.analysis.primary_contrasts['g8-ppo']);
    const c = analysisModel(d).contrasts[0];
    assert.equal(c.stats, null);
    assert.equal(c.equivalent, false);
  }
});

test('exploratory contrasts retain raw inference without invented adjusted tests', () => {
  const d = fixture(), c = d.analysis.exploratory_contrasts['g2-ppo'];
  delete c.holm_p_t; delete c.holm_p_signflip; delete c.holm_tost_p;
  const model = analysisModel(d).contrasts[3];
  assert.equal(model.stats.p, 1);
  assert.equal(model.equivalent, false);
});


test('pair duplication, forged summaries, and out-of-plan cohorts fail closed', () => {
  const changes = [
    (d, c) => { c.pairs[1] = { ...c.pairs[0] }; },
    (d, c) => { c.pairs[1] = { ...c.pairs[0] }; c.seeds[1] = 1; },
    (d, c) => { c.diffs[0] = .01; },
    (d, c) => { c.mean = .001; },
    (d, c) => { c.t_ci = [-.01, .01]; },
    (d, c) => { c.tost_ci90 = [-.01, .01]; },
    (d, c) => { c.tost_p = .001; },
    (d, c) => { c.holm_tost_p = .001; },
    (d) => { d.analysis.plan.planned_seeds = [3, 4]; },
    (d) => { d.samestack.runs[0].manifest.analysis_plan_id = 'other'; },
    (d) => { d.analysis.plan.plan_id = 'other'; },
  ];
  for (const change of changes) {
    const d = fixture(); change(d, d.analysis.primary_contrasts['g8-ppo']);
    const model = analysisModel(d);
    assert.equal(model.contrasts[0].stats, null);
    assert.equal(model.contrasts[0].equivalent, false);
  }
});
