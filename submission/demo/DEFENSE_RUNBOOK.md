# Defense Runbook

## Before entering the room

```bash
./submission/demo/demo.sh --self-test
./submission/demo/demo.sh
```

Confirm both commands exit 0 and the second ends with `DEMO STATUS: PASS`. Open `submission/demo/output/demo_report.html` directly in a browser and keep it open. This file works without a server, Wi-Fi, GPU, API key, package install, or access to the live Hugging Face Space.

Keep a terminal at the repository root. If the evaluator asks for a fresh run:

```bash
./submission/demo/demo.sh --serve
```

Open `http://127.0.0.1:8765/demo_report.html` and stop the server with Ctrl-C after the demonstration.

## 90-second narration

**0–20 seconds — scope.** “This is a deterministic mechanism and artifact-integrity demonstration. It does not claim to retrain the model or reproduce a benchmark on this laptop.”

**20–50 seconds — group-relative signal.** Point to “Mixed outcomes.” “For each prompt, GRPO compares rewards within the sampled group. Subtracting the group mean and dividing by population standard deviation gives negative advantages to below-mean samples and positive advantages to above-mean samples.” Point to the two equal-reward rows. “All-wrong and all-correct groups both have zero within-group variance, so the relative advantages are zero. ZVF counts how often that happens; it is a signal-availability diagnostic, not a performance metric.”

**50–75 seconds — artifact audit.** Point to the SHA-256 and the 80 rewards. “The second half checks the exact recorded artifact. It recomputes every per-problem mean, every zero-variance indicator, the overall reward mean, and overall ZVF. The SHA binds the display to the reviewed input bytes.”

**75–90 seconds — honest boundary.** Point to “Claims intentionally excluded.” “A pass proves the implementation and artifact arithmetic are inspectable. It does not prove generalization, causal improvement, or state-of-the-art performance. Full training remains a separate compute-heavy reproduction.”

## Likely evaluator questions

**Why is standard deviation zero for all-correct as well as all-wrong?** Relative optimization needs differences inside the group. Equal rewards provide none, regardless of reward level.

**Why use population standard deviation?** The training implementation in this repository normalizes by the square root of the mean squared deviation over the sampled group. The demo mirrors that convention.

**Does ZVF prove training collapse?** No. High ZVF can reflect incapacity or saturation. The dashboard calls it a signal-availability/triage diagnostic and does not infer future performance.

**Why not run full Tinker training live?** It requires external credentials, remote scheduling, time, and stochastic sampling. The offline demo isolates the deterministic mechanism and artifact contract that an evaluator can verify during a defense.

**What does 68.75% mean here?** It is the recomputed mean of the 80 recorded binary rewards in one project artifact. It is not presented as held-out benchmark accuracy or a new result.

## Failure fallback

1. If port 8765 is busy, run `./submission/demo/demo.sh --serve --port 8877`.
2. If the local server cannot start, open `submission/demo/output/demo_report.html` directly.
3. If the artifact SHA check fails, do not bypass it. Show the failure, run `git diff -- experiments/results/tinker_direct_eval.json`, and explain that the input bytes no longer match the reviewed contract.
4. If Groq live mode fails, return to `./submission/demo/demo.sh`. Live mode is optional and supports no thesis claim.

