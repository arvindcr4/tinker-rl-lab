# E9 research directions

Recorded: 2026-08-29T04:16:16Z

These are prospective, testable directions. None is a measured improvement or completed benchmark result.

## 1. Provenance-locked recovery of archived programs

**Hypothesis:** Replaying already-sampled programs against the pinned native benchmark can recover missing terminal outcomes without generation spend.

**Design:** Require exact competition ID, pinned MLE-bench commit, immutable model/export commit, W&B run URL, source-receipt hash, source-run ID, and solution hash before execution. Record replay mode and charge as separate receipt fields. Do not modify the program in the pure replay arm.

**Measure:** submission existence, native-grade success, failure class, elapsed time, and incremental bridge charge. Any competition score is reported only from the native grader.

**Current signal:** The first Russian replay validated provenance and cost isolation but exposed a float/string bug. A labeled pass-through repair advanced execution to a missing submission-ID bug, and the schema-aligned replay then produced a hash-locked submission. A native-only regrade under pinned `pandas==2.2.2` recovered a valid `0.97348` competition score with no medal and no model-generation charge. This completes the recovery experiment for one competition; it is not the E9 suite score.

## 2. Exact-adapter inference continuity

**Hypothesis:** The existing immutable merged checkpoint can provide a controlled inference path after the Tinker sampler checkpoint disappeared.

**Current preflight evidence:** The adapter header is PEFT-style 3D for fused MoE expert tensors, and an earlier receipt records a complete 862-tensor merge into a 26-shard checkpoint with online W&B provenance. Prefer validating that already-merged checkpoint over creating a second merge or an untested dynamic-LoRA path.

**Preflight before paid GPU work:**

1. Verify the merged pointer and all 26 shard hashes before GPU load.
2. Pin the base model revision, adapter commit, merge receipt, vLLM version, tokenizer, dtype, and serving flags.
3. Start online W&B before any GPU/inference work and attach immutable Hugging Face receipts.
4. Run a tiny deterministic prompt canary and compare repeatability and output validity; do not claim equivalence to the deleted Tinker sampler checkpoint.
5. Meter all work against the authorized persistent ceiling of $55.91445263.

**Stop rule:** Do not deploy if layout compatibility is ambiguous, the monitoring/checkpoint receipts are missing, or the projected authorized spend would be exceeded.

## 3. Program reliability gates

**Hypothesis:** Cheap static and sandboxed pre-submission checks can reduce terminal failures without leaking private-test feedback.

**Candidate gates:** parse/extract executable Python, reject prose-only outputs, run `py_compile`, inspect train/test/sample schemas, sample representative null and mixed-type values, validate required output columns and row count, and execute a short resource-bounded smoke test.

**Evaluation:** Pre-register the gates, apply them uniformly, and compare valid-submission rate and failure taxonomy on a held-out set of competitions. Keep the agent's generated program immutable after the gate in the primary arm.

## 4. Benchmark-integrity and version audit

**Hypothesis:** Version pinning plus contamination checks will explain a meaningful subset of null outcomes and prevent invalid comparisons.

**Design:** Record the exact MLE-bench commit, task data hashes, environment lock, grader version, and known-issue applicability for every run. Run the official plagiarism/contamination checks where licensed and available. Preserve the private-test boundary.

**Reporting:** Separate agent-execution failure, invalid submission, grader/environment failure, provider/data blocker, and valid native grade. Do not compare against a paused or different-version leaderboard without an explicit version caveat.

## Recommended evaluation contract

- Primary: official Any Medal (%) across all 75 competitions; mean and SEM over at least three seeds.
- Stratification: Low, Medium, High, and All complexity groups.
- Secondary: native-grade coverage, valid-submission rate, failure taxonomy, wall time, and actual cost.
- Claim rule: partial runs, repaired replays, public samples, and fixtures remain clearly labeled and cannot be promoted to the full-suite score.
