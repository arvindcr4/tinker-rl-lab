// Read-only audit of immutable historical reward scores under the declared strict
// grader. Disagreements are evidence of a changed contract, never a reason to
// rewrite historical scores or loosen the grader.
// Usage: node tests/grade_all.mjs [--json] [--report-only]
// Exit 1: score disagreement; exit 2: incomplete/invalid input coverage.
import { readFileSync, readdirSync, existsSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { fileURLToPath, pathToFileURL } from "node:url";
import { grade, toNum, GRADING_SPEC_ID } from "../src/grade.js";

const ROOT = fileURLToPath(new URL("../../../../", import.meta.url));

// Python's legacy JSON writer emitted bare nonfinite tokens. Replace only
// tokens outside strings; a global /NaN/ replacement would corrupt user text.
export function parseLegacyJson(raw) {
  let normalized = "", inString = false, escaped = false, nonfiniteTokens = 0;
  for (let i = 0; i < raw.length; i++) {
    const char = raw[i];
    if (inString) {
      normalized += char;
      if (escaped) escaped = false;
      else if (char === "\\") escaped = true;
      else if (char === '"') inString = false;
      continue;
    }
    if (char === '"') { inString = true; normalized += char; continue; }
    const token = /^(?:-?Infinity|NaN)(?=$|[\s,\]}])/.exec(raw.slice(i));
    if (token && (i === 0 || /[\s:,\[]/.test(raw[i - 1]))) {
      normalized += "null";
      nonfiniteTokens++;
      i += token[0].length - 1;
    } else normalized += char;
  }
  return { value: JSON.parse(normalized), nonfiniteTokens };
}

// Diagnostic attribution only: never changes strict grading or stored rewards.
// Exact equality after historical formatting removal distinguishes a convention
// change from a genuinely different numeric answer.
function disagreementCause(result, gold) {
  if (result.correct === 1) return "historical_false_negative";
  if (result.boxed === null) return "missing_or_malformed_box";
  if (result.status === "ok") return "numeric_mismatch";
  const cleaned = result.boxed.replace(/\\text\{[^}]*\}|\\mathrm\{[^}]*\}|\\[,;:! ]|\\%|%|\\\$|\$|,|\s/g, "").replace(/\.+$/, "");
  const candidate = toNum(cleaned), expected = toNum(gold);
  if (candidate && expected && candidate.numerator === expected.numerator && candidate.denominator === expected.denominator) {
    const types = [
      [/\$/, "currency"], [/%/, "percent"], [/\\(?:text|mathrm)\{/, "units"],
    ].filter(([pattern]) => pattern.test(result.boxed)).map(([, name]) => name);
    return `strict_syntax_${types.length === 1 ? types[0] : "other"}`;
  }
  return "invalid_numeric_or_unsupported_syntax";
}

const count = (object, key) => { object[key] = (object[key] || 0) + 1; };

export function auditStoredScores({ resultsDir, goldPath }) {
  const gold = JSON.parse(readFileSync(goldPath, "utf8")).test
    .map((t) => t.answer.split("####").pop().trim());
  if (gold.some((answer) => toNum(answer) === null)) throw new Error("gold answers must be finite exact numbers with valid grouping");
  const files = readdirSync(resultsDir).filter((f) => /_s\d+\.json$/.test(f)).sort();
  const report = {
    grading_spec_id: GRADING_SPEC_ID,
    historical_scores_mutated: false,
    completions: 0, runs: files.length, runs_checked: 0, expected_items_per_phase: gold.length,
    disagreements: 0, stored_correct_now_incorrect: 0, stored_incorrect_now_correct: 0,
    missing_sidecars: [], invalid_inputs: [], length_mismatches: [], invalid_scores: [],
    legacy_nonfinite_tokens: 0, examples: [], by_run: [],
    cause_counts: { numeric_mismatch: 0 }, cause_examples: {},
    by_phase: { pre: { disagreements: 0, cause_counts: {} }, post: { disagreements: 0, cause_counts: {} } },
  };
  for (const file of files) {
    const runPath = new URL(file, pathToFileURL(resultsDir.endsWith("/") ? resultsDir : resultsDir + "/"));
    const sidecarPath = new URL(file.replace(/\.json$/, "_texts.json.gz"), runPath);
    if (!existsSync(sidecarPath)) { report.missing_sidecars.push(file); continue; }
    let run, texts;
    try {
      const parsed = parseLegacyJson(readFileSync(runPath, "utf8"));
      run = parsed.value;
      report.legacy_nonfinite_tokens += parsed.nonfiniteTokens;
      texts = JSON.parse(gunzipSync(readFileSync(sidecarPath)).toString("utf8"));
    } catch (error) {
      report.invalid_inputs.push({ file, reason: error.message });
      continue;
    }
    report.runs_checked++;
    const row = { file, completions: 0, disagreements: 0, stored_correct_now_incorrect: 0, stored_incorrect_now_correct: 0,
      by_phase: { pre: { disagreements: 0, cause_counts: {} }, post: { disagreements: 0, cause_counts: {} } } };
    for (const phase of ["pre", "post"]) {
      const stored = run[`${phase}_correct`], completions = texts[phase];
      if (!Array.isArray(stored) || !Array.isArray(completions)) {
        report.invalid_inputs.push({ file, phase, reason: "missing score or text array" });
        continue;
      }
      if (stored.length !== gold.length || completions.length !== gold.length) {
        report.length_mismatches.push({ file, phase, gold: gold.length, scores: stored.length, texts: completions.length });
      }
      for (let index = 0; index < Math.min(gold.length, stored.length, completions.length); index++) {
        if (![0, 1].includes(stored[index]) || typeof completions[index] !== "string") {
          report.invalid_scores.push({ file, phase, index, reason: "expected binary stored score and string completion" });
          continue;
        }
        report.completions++;
        row.completions++;
        const result = grade(completions[index], gold[index]);
        if (result.correct === stored[index]) continue;
        report.disagreements++;
        row.disagreements++;
        const direction = stored[index] ? "stored_correct_now_incorrect" : "stored_incorrect_now_correct";
        report[direction]++;
        row[direction]++;
        const cause = disagreementCause(result, gold[index]);
        count(report.cause_counts, cause);
        for (const target of [report.by_phase[phase], row.by_phase[phase]]) {
          target.disagreements++;
          count(target.cause_counts, cause);
        }
        const example = { file, phase, index, strict: result.correct, stored: stored[index],
          gold: gold[index], boxed: result.boxed?.slice(0, 200) ?? null, status: result.status, cause };
        if (report.examples.length < 10) report.examples.push(example);
        const causeExamples = report.cause_examples[cause] ??= [];
        if (causeExamples.length < 3 && !causeExamples.some((e) => e.boxed === example.boxed && e.gold === example.gold)) causeExamples.push(example);
      }
    }
    report.by_run.push(row);
  }
  report.coverage_complete = report.runs > 0 && report.missing_sidecars.length === 0 && report.invalid_inputs.length === 0
    && report.length_mismatches.length === 0 && report.invalid_scores.length === 0;
  report.format_convention_disagreements = Object.entries(report.cause_counts)
    .filter(([cause]) => cause.startsWith("strict_syntax_")).reduce((sum, [, total]) => sum + total, 0);
  report.numeric_mismatch_disagreements = report.cause_counts.numeric_mismatch;
  return report;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  try {
    const option = (name, fallback) => {
      const index = process.argv.indexOf(name);
      if (index < 0) return fallback;
      if (!process.argv[index + 1] || process.argv[index + 1].startsWith("--")) throw new Error(`${name} needs a path`);
      return process.argv[index + 1];
    };
    const report = auditStoredScores({
      resultsDir: option("--results-dir", ROOT + "platform_hybrid/experiments/results/samestack_gsm8k_v2s/"),
      goldPath: option("--gold-path", ROOT + "reports/project_finish_2026-10-11/validation/samestack_v2/gsm8k_test_gold.json"),
    });
    if (process.argv.includes("--json")) console.log(JSON.stringify(report));
    else {
      console.log(`graded ${report.completions} stored completions from ${report.runs_checked}/${report.runs} runs under ${report.grading_spec_id}`);
      console.log(`${report.disagreements} disagree with historical scores (${report.stored_correct_now_incorrect} correct→incorrect, ${report.stored_incorrect_now_correct} incorrect→correct); historical files unchanged`);
      console.log(`causes: ${JSON.stringify(report.cause_counts)}; pre=${report.by_phase.pre.disagreements}, post=${report.by_phase.post.disagreements}`);
      console.log(`coverage ${report.coverage_complete ? "complete" : "INCOMPLETE"}: ${report.missing_sidecars.length} missing sidecars, ${report.invalid_inputs.length} invalid inputs, ${report.length_mismatches.length} length mismatches, ${report.invalid_scores.length} invalid scores`);
      for (const example of report.examples) console.log(JSON.stringify(example));
    }
    process.exitCode = !report.coverage_complete ? 2 : report.disagreements && !process.argv.includes("--report-only") ? 1 : 0;
  } catch (error) {
    console.error(`Historical grading audit could not complete: ${error.message}`);
    process.exitCode = 2;
  }
}
