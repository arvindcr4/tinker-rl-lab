// Exhaustive check: the in-page grader must reproduce the stored score for every stored completion.
// Usage: node tests/grade_all.mjs [--json]   (exit 1 on any disagreement)
import { readFileSync, readdirSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { grade } from "../src/grade.js";

const ROOT = new URL("../../../../", import.meta.url).pathname;
const V2 = ROOT + "platform_hybrid/experiments/results/samestack_gsm8k_v2s/";
const gold = JSON.parse(readFileSync(ROOT + "reports/project_finish_2026-10-11/validation/samestack_v2/gsm8k_test_gold.json"))
  .test.map((t) => t.answer.split("####").pop().trim().replaceAll(",", ""));

let n = 0, bad = 0;
for (const f of readdirSync(V2).filter((f) => /_s\d+\.json$/.test(f)).sort()) {
  const run = JSON.parse(String(readFileSync(V2 + f)).replace(/\bNaN\b/g, "null"));  // Python json writes bare NaN
  const tx = JSON.parse(gunzipSync(readFileSync(V2 + f.replace(".json", "_texts.json.gz"))));
  for (const [k, stored] of [["pre", run.pre_correct], ["post", run.post_correct]]) {
    tx[k].forEach((t, i) => {
      n++;
      const js = grade(t, gold[i]).correct;
      if (js !== stored[i] && ++bad <= 5) console.error(`${f} ${k}[${i}] js=${js} stored=${stored[i]}`);
    });
  }
}
const runs = readdirSync(V2).filter((f) => /_s\d+\.json$/.test(f)).length;
if (process.argv.includes("--json")) console.log(JSON.stringify({ completions: n, runs, disagreements: bad }));
else console.log(`graded ${n} stored completions from ${runs} runs; ${bad} disagree with stored scores`);
process.exit(bad ? 1 : 0);
