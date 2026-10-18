import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync, writeFileSync, mkdtempSync, mkdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";
import { gzipSync } from "node:zlib";
import { grade, lastBoxed, toNum, GRADING_SPEC_ID } from "../src/grade.js";
import { paired, signflip, holm, tcdf, tppf, pct, sgn, pfmt, MAX_EXACT_SIGNFLIP_N } from "../src/stats.js";
import { auditStoredScores, parseLegacyJson } from "./grade_all.mjs";

const ROOT = fileURLToPath(new URL("../../../../", import.meta.url));
const gradingFixture = JSON.parse(readFileSync(new URL("../../../../tests/fixtures/samestack_grading_v2.json", import.meta.url)));
const statisticsFixture = JSON.parse(readFileSync(new URL("./fixtures/stats_parity.json", import.meta.url)));
const close = (actual, expected, label = "value") => assert.ok(Math.abs(actual - expected) <= 2e-10 * Math.max(1, Math.abs(expected)), `${label}: ${actual} != ${expected}`);

function python(script, input) {
  const result = spawnSync(process.env.PYTHON || "python3", ["-c", script, ROOT], {
    input: JSON.stringify(input), encoding: "utf8", env: { ...process.env, PYTHONDONTWRITEBYTECODE: "1" },
  });
  assert.equal(result.status, 0, `Python parity check failed: ${result.stderr}`);
  return JSON.parse(result.stdout);
}

test("strict grader satisfies the versioned shared fixture", () => {
  assert.equal(gradingFixture.spec_id, GRADING_SPEC_ID);
  for (const row of gradingFixture.cases) {
    const result = grade(row.text, row.gold);
    assert.equal(result.correct, row.reward, JSON.stringify(row));
    if (row.status) {
      assert.equal(result.status, row.status, JSON.stringify(row));
      assert.equal(toNum(result.boxed), null);
      assert.equal(grade("\\boxed{42}", result.boxed).status, "invalid gold");
    }
  }
});

test("boxes require the actual command, immediate optional whitespace and balanced braces", () => {
  for (const value of ["\\boxedX{42}", "\\boxed99 unrelated text {42}", "boxed{42}", "\\boxed{42", "\\boxed {\\frac{42}{1}"]) {
    assert.equal(lastBoxed(value), null, value);
    assert.equal(grade(value, "42").correct, 0);
  }
  assert.equal(lastBoxed("\\boxed \n {\\frac{1}{2}}"), "\\frac{1}{2}");
  assert.equal(lastBoxed("\\boxed{42} followed by \\boxed99 unrelated {0}"), "42");
  assert.equal(lastBoxed("\\boxed{42} followed by \\boxed{unclosed"), "42");
  assert.equal(grade("\\boxed{42} then \\boxed{bad}", "42").correct, 0);
  assert.equal(lastBoxed(null), null);
});

test("grading compares exact rationals without float precision loss or six-digit rounding", () => {
  const good = [
    ["+4.2e1", "42"], ["-\\tfrac{3}{-2}", "1.5"], ["1.5/3", "0.5"],
    ["\\frac{1}{3}", "2/6"], ["9007199254740993", "9007199254740993"],
    ["1e1000", "10e999"], ["1e-1000", "0.1e-999"], ["1\\,234", "1234"],
    ["\\$1,234.00", "1234"], ["10\\%", "10"], ["42 seconds", "42"], ["42\\text{ second }", "42"],
  ];
  for (const [answer, gold] of good) assert.equal(grade(`\\boxed{${answer}}`, gold).correct, 1, `${answer} vs ${gold}`);
  for (const [answer, gold] of [["9007199254740993", "9007199254740992"], ["42.0000001", "42"], ["1.0000000000000000001", "1"], ["0.333333", "1/3"]]) {
    assert.equal(grade(`\\boxed{${answer}}`, gold).correct, 0, `${answer} vs ${gold}`);
  }
  assert.deepEqual(toNum("2/4"), { numerator: 1n, denominator: 2n });
  for (const value of ["1/0", "\\frac{1}{0}", "NaN", "Infinity", "1_000", "12,34", "42 apples", "42\\text{ apples}", "1 2", "1e10001", "\\$42%", "42\\text{seconds 99}", "42\\text{or forty three}", "\\frac{50%}{100}", "42% seconds", "$$42", "42%%"]) {
    assert.equal(toNum(value), null, value);
  }
  assert.equal(grade("\\boxed{42}", null).status, "invalid gold");
});

test("JavaScript grader agrees with independent analysis and training implementations", () => {
  const actual = python(`
import importlib.util, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
protocol = load('samestack_protocol', root / 'platform_hybrid/experiments/samestack_protocol.py')
analysis_dir = root / 'reports/project_finish_2026-10-11/validation/samestack_v2'
sys.path.insert(0, str(analysis_dir))
audit = load('v2_analysis_parity', analysis_dir / 'v2_analysis.py')
cases = json.load(sys.stdin)
assert protocol.GRADING_SPEC_ID == audit.GRADING_SPEC_ID == 'boxed-exact-rational-v2'
print(json.dumps([[protocol.boxed_reward(c['text'], c['gold']), audit.grade(c['text'], c['gold'])[0]] for c in cases]))
`, gradingFixture.cases);
  gradingFixture.cases.forEach((row, index) => assert.deepEqual(actual[index], [row.reward, row.reward], JSON.stringify(row)));
});

test("paired and sign-flip reject missing, singleton and nonfinite inputs", () => {
  for (const input of [[], [0], [0, NaN], [0, Infinity], [-Infinity, 0], [0, "1"], new Array(2), null, "01"]) {
    assert.throws(() => paired(input), undefined, String(input));
    assert.throws(() => signflip(input), undefined, String(input));
  }
  for (const margin of [0, -0.02, NaN, Infinity, null, "0.02"]) assert.throws(() => paired([0, 1], margin));
});

test("constant pairs and TOST boundaries produce defined p-values", () => {
  for (const row of statisticsFixture.cases) {
    const result = paired(row.differences);
    for (const field of ["p", "tost", "sf"]) if (field in row) close(result[field], row[field], row.name + " " + field);
    if ("positive5ppP" in row) close(result.positive_5pp_test.p_value, row.positive5ppP, row.name);
    for (const field of ["md", "sd", "se", "lo", "hi", "p", "tost", "sf", "tost_lower_p", "tost_upper_p"]) assert.ok(Number.isFinite(result[field]), row.name + " " + field);
  }
});

test("difference 95% and equivalence 90% intervals and directional +5pp test are labeled", () => {
  const result = paired([-0.01, 0, 0.01, 0.02, 0.03]);
  assert.deepEqual(result.difference_ci_95, [result.lo, result.hi]);
  assert.ok(result.equivalence_ci_90[0] > result.lo);
  assert.ok(result.equivalence_ci_90[1] < result.hi);
  assert.equal(result.equivalence_margin, 0.02);
  assert.equal(result.positive_5pp_test.null_hypothesis, "mean >= +0.05");
  assert.equal(result.positive_5pp_test.alternative_hypothesis, "mean < +0.05");
  assert.equal(result.positive_5pp_test.upper_bound_95, result.equivalence_ci_90[1]);
  assert.equal(result.positive_5pp_test.reject_at_0_05, result.positive_5pp_test.p_value < 0.05);
});

test("exact sign-flip has a hard bounded size, no 32-bit overflow and no tiny-value tie inflation", () => {
  assert.equal(MAX_EXACT_SIGNFLIP_N, 20);
  for (const n of [21, 31, 32, 40]) {
    assert.throws(() => signflip(Array(n).fill(1)), /limited/);
    const result = paired(Array(n).fill(0.01));
    assert.equal(result.sf, null);
    assert.equal(result.sf_available, false);
    assert.match(result.sf_unavailable_reason, /20/);
  }
  assert.equal(signflip(Array(20).fill(1)), 2 / 2 ** 20);
  assert.equal(signflip([1e-20, 2e-20, 3e-20, 4e-20, 5e-20]), 2 / 2 ** 5);
});

test("Holm rejects invalid probabilities and preserves monotonic step-down adjustment", () => {
  assert.deepEqual(holm({ a: 0.01, b: 0.04, c: 0.03 }), { a: 0.03, c: 0.06, b: 0.06 });
  assert.deepEqual(holm({ a: 0, b: 1 }), { a: 0, b: 1 });
  assert.deepEqual(holm({}), {});
  for (const value of [NaN, Infinity, -0.01, 1.01, null, "0.1", true]) assert.throws(() => holm({ a: value }));
  for (const value of [null, [0.1], 0.1]) assert.throws(() => holm(value));
});

test("Student t helpers validate domains and do not clip large quantiles at 60", () => {
  close(tcdf(0, 1), 0.5);
  close(tppf(0.975, 1), 12.706204736432095);
  close(tppf(0.999, 1), 318.3088389855419);
  close(tppf(1e-20, 1), -1 / (Math.PI * 1e-20));
  assert.equal(tppf(0, 1), -Infinity);
  assert.equal(tppf(1, 1), Infinity);
  for (const df of [0, -1, NaN, Infinity]) assert.throws(() => tcdf(1, df));
  for (const q of [-0.1, 1.1, NaN, null]) assert.throws(() => tppf(q, 1));
  assert.throws(() => tcdf(NaN, 1));
});

test("JavaScript statistics match the independent Python analysis on the shared fixture", () => {
  const references = python(`
import json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(sys.argv[1]) / 'reports/project_finish_2026-10-11/validation/samestack_v2'))
from stats_lib import paired
rows = json.load(sys.stdin)
print(json.dumps([paired(dict(enumerate(row['differences'])), dict(enumerate([0.] * len(row['differences'])))) for row in rows], allow_nan=False))
`, statisticsFixture.cases);
  statisticsFixture.cases.forEach((row, index) => {
    const actual = paired(row.differences), reference = references[index];
    for (const [js, py] of [["md", "mean"], ["sd", "sd"], ["p", "p_t"], ["sf", "p_signflip"], ["tost", "tost_p"], ["tost_lower_p", "tost_lower_p"], ["tost_upper_p", "tost_upper_p"]]) close(actual[js], reference[py], row.name + " " + js);
    actual.difference_ci_95.forEach((value, i) => close(value, reference.t_ci[i], row.name + " CI95"));
    actual.equivalence_ci_90.forEach((value, i) => close(value, reference.tost_ci90[i], row.name + " CI90"));
    close(actual.positive_5pp_test.p_value, reference.positive_5pp_test.p_value, row.name + " positive5pp");
  });
});

test("Python and JavaScript both reject invalid inference inputs and oversized exact enumeration", () => {
  const results = python(`
import json, math, pathlib, sys
sys.path.insert(0, str(pathlib.Path(sys.argv[1]) / 'reports/project_finish_2026-10-11/validation/samestack_v2'))
from stats_lib import paired, signflip, holm
def rejected(fn):
    try: fn()
    except (ValueError, TypeError): return True
    return False
invalid = [[], [0.], [0., math.nan], [0., math.inf], [-math.inf, 0.]]
output = []
for d in invalid:
    output.append(rejected(lambda: paired(dict(enumerate(d)), dict(enumerate([0.] * len(d))))))
    output.append(rejected(lambda: signflip(d)))
for n in [21, 31, 32, 40]:
    output.append(rejected(lambda: signflip([1.] * n)))
for value in [math.nan, math.inf, -.01, 1.01, None, '0.1', True]:
    output.append(rejected(lambda: holm({'a': value})))
print(json.dumps(output))
`, null);
  const actual = [];
  const rejected = (fn) => { try { fn(); return false; } catch { return true; } };
  for (const d of [[], [0], [0, NaN], [0, Infinity], [-Infinity, 0]]) {
    actual.push(rejected(() => paired(d)), rejected(() => signflip(d)));
  }
  for (const n of [21, 31, 32, 40]) actual.push(rejected(() => signflip(Array(n).fill(1))));
  for (const value of [NaN, Infinity, -0.01, 1.01, null, "0.1", true]) actual.push(rejected(() => holm({ a: value })));
  assert.deepEqual(actual, results);
  assert.ok(results.every(Boolean));
});

test("number formatters render missing or nonfinite values as unavailable", () => {
  for (const format of [pct, sgn, pfmt]) for (const value of [null, undefined, NaN, Infinity, -Infinity, "0"]) assert.equal(format(value), "unavailable");
  assert.equal(pct(0), "0.0%");
  assert.equal(sgn(0), "+0.0000");
});

test("legacy JSON nonfinite token handling leaves quoted text unchanged", () => {
  const parsed = parseLegacyJson('{"a": NaN, "b": Infinity, "c": -Infinity, "text": "NaN Infinity -Infinity", "escape": "\\\"NaN"}');
  assert.equal(parsed.nonfiniteTokens, 3);
  assert.deepEqual(parsed.value, { a: null, b: null, c: null, text: "NaN Infinity -Infinity", escape: '"NaN' });
  assert.throws(() => parseLegacyJson('{"a": NaNfoo}'));
});

test("historical audit reports disagreement directions and incomplete coverage without changing scores", () => {
  const directory = mkdtempSync(join(tmpdir(), "samestack-grading-"));
  try {
    const goldPath = join(directory, "gold.json");
    writeFileSync(goldPath, JSON.stringify({ test: [{ answer: "#### 42" }] }));
    const original = '{"pre_correct":[1],"post_correct":[0],"legacy":NaN,"text":"NaN"}';
    writeFileSync(join(directory, "demo_s1.json"), original);
    writeFileSync(join(directory, "demo_s1_texts.json.gz"), gzipSync(JSON.stringify({ pre: ["\\boxedX{42}"], post: ["\\boxed{42}"] })));
    const report = auditStoredScores({ resultsDir: directory, goldPath });
    assert.equal(report.disagreements, 2);
    assert.equal(report.stored_correct_now_incorrect, 1);
    assert.equal(report.stored_incorrect_now_correct, 1);
    assert.equal(report.completions, 2);
    assert.equal(report.coverage_complete, true);
    assert.equal(report.legacy_nonfinite_tokens, 1);
    assert.equal(readFileSync(join(directory, "demo_s1.json"), "utf8"), original);
    writeFileSync(join(directory, "demo_s2.json"), original);
    const incomplete = auditStoredScores({ resultsDir: directory, goldPath });
    assert.deepEqual(incomplete.missing_sidecars, ["demo_s2.json"]);
    assert.equal(incomplete.coverage_complete, false);
  } finally { rmSync(directory, { recursive: true, force: true }); }
});

test("historical audit retains valid GSM8K decorations while exposing actual numeric mismatches", () => {
  const directory = mkdtempSync(join(tmpdir(), "samestack-causes-"));
  try {
    const goldPath = join(directory, "gold.json");
    const answers = ["\\boxed{\\$42}", "\\boxed{42\\%}", "\\boxed{42\\text{seconds}}", "\\boxed{42.0000001}"];
    writeFileSync(goldPath, JSON.stringify({ test: answers.map(() => ({ answer: "#### 42" })) }));
    writeFileSync(join(directory, "demo_s1.json"), JSON.stringify({ pre_correct: [1, 1, 1, 1], post_correct: [1, 1, 1, 1] }));
    writeFileSync(join(directory, "demo_s1_texts.json.gz"), gzipSync(JSON.stringify({ pre: answers, post: answers.map(() => "\\boxed{42}") })));
    const report = auditStoredScores({ resultsDir: directory, goldPath });
    assert.deepEqual(report.cause_counts, { numeric_mismatch: 1 });
    assert.equal(report.format_convention_disagreements, 0);
    assert.equal(report.numeric_mismatch_disagreements, 1);
    assert.equal(report.by_phase.pre.disagreements, 1);
    assert.equal(report.by_phase.post.disagreements, 0);
    assert.equal(report.by_run[0].by_phase.pre.cause_counts.numeric_mismatch, 1);
    assert.equal(report.cause_examples.numeric_mismatch[0].boxed, "42.0000001");
  } finally { rmSync(directory, { recursive: true, force: true }); }
});

test("historical audit validates grouping in both completions and unmodified gold", () => {
  const directory = mkdtempSync(join(tmpdir(), "samestack-grouping-"));
  try {
    const goldPath = join(directory, "gold.json");
    const runPath = join(directory, "demo_s1.json");
    const original = JSON.stringify({ pre_correct: [1], post_correct: [1] });
    writeFileSync(runPath, original);
    writeFileSync(goldPath, JSON.stringify({ test: [{ answer: "#### 42" }] }));
    writeFileSync(join(directory, "demo_s1_texts.json.gz"), gzipSync(JSON.stringify({ pre: ["\\boxed{42.0,000}"], post: ["\\boxed{42}"] })));
    const report = auditStoredScores({ resultsDir: directory, goldPath });
    assert.equal(report.disagreements, 1);
    assert.equal(report.stored_correct_now_incorrect, 1);
    assert.equal(report.examples[0].status, "unparsable");
    writeFileSync(goldPath, JSON.stringify({ test: [{ answer: "#### 42.0,000" }] }));
    assert.throws(() => auditStoredScores({ resultsDir: directory, goldPath }), /gold answers.*valid grouping/);
    writeFileSync(goldPath, JSON.stringify({ test: [{ answer: "#### 1,234" }] }));
    writeFileSync(join(directory, "demo_s1_texts.json.gz"), gzipSync(JSON.stringify({ pre: ["\\boxed{1234}"], post: ["\\boxed{1,234.0}"] })));
    assert.equal(auditStoredScores({ resultsDir: directory, goldPath }).disagreements, 0);
    assert.equal(readFileSync(runPath, "utf8"), original);
  } finally { rmSync(directory, { recursive: true, force: true }); }
});

test("audit rejects empty gold, malformed records, unequal arrays and declared denominator drift", () => {
  const directory = mkdtempSync(join(tmpdir(), "samestack-alignment-"));
  try {
    const goldPath = join(directory, "gold.json");
    const runPath = join(directory, "demo_s1.json");
    const sidecar = join(directory, "demo_s1_texts.json.gz");
    const audit = () => auditStoredScores({ resultsDir: directory, goldPath });
    writeFileSync(goldPath, JSON.stringify({ test: [] }));
    assert.throws(audit, /nonempty/);
    writeFileSync(goldPath, JSON.stringify({ test: [{ answer: "#### 42" }] }));
    for (const [record, texts] of [
      [null, {}], [{}, null],
      [{ pre_correct: [], post_correct: [1] }, { pre: ["\\boxed{42}"], post: ["\\boxed{42}"] }],
      [{ pre_correct: [1], post_correct: [1] }, { pre: ["\\boxed{42}", "extra"], post: ["\\boxed{42}"] }],
      [{ n_eval: 2, pre_correct: [1], post_correct: [1] }, { pre: ["\\boxed{42}"], post: ["\\boxed{42}"] }],
      [{ pre_correct: [1], post_correct: [2] }, { pre: ["\\boxed{42}"], post: ["\\boxed{42}"] }],
    ]) {
      writeFileSync(runPath, JSON.stringify(record));
      writeFileSync(sidecar, gzipSync(JSON.stringify(texts)));
      const result = audit();
      assert.equal(result.coverage_complete, false, JSON.stringify({ record, texts }));
      assert.ok(result.invalid_inputs.length + result.length_mismatches.length + result.invalid_scores.length > 0);
    }
  } finally { rmSync(directory, { recursive: true, force: true }); }
});


test("grading audit discovers nested promoted run sidecars", () => {
  const root = mkdtempSync(join(tmpdir(), "nested-grading-"));
  try {
    const leaf = join(root, "grpo_g8_s42"); mkdirSync(leaf);
    const goldPath = join(root, "gold.json");
    writeFileSync(goldPath, JSON.stringify({ test: [{ answer: "#### 1" }] }));
    writeFileSync(join(leaf, "grpo_g8_s42.json"), JSON.stringify({ n_eval: 1, pre_correct: [1], post_correct: [1] }));
    writeFileSync(join(leaf, "grpo_g8_s42_texts.json.gz"), gzipSync(JSON.stringify({ pre: ["\\boxed{1}"], post: ["\\boxed{1}"] })));
    const report = auditStoredScores({ resultsDir: root, goldPath });
    assert.equal(report.runs_checked, 1);
    assert.equal(report.completions, 2);
    assert.equal(report.coverage_complete, true);
    assert.equal(report.disagreements, 0);
  } finally { rmSync(root, { recursive: true, force: true }); }
});
