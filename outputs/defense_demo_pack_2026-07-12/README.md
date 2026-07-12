# Defense demo pack — 12 July 2026

Open `index.html` first. It provides the four-tier failover ladder in presentation order.

## 1. Live interactive demo

- Space: https://arvindcr4-tinkerrl-bench-demo.hf.space
- Hub: https://huggingface.co/spaces/arvindcr4/tinkerrl-bench-demo
- Deployed revision: `73c2daf213fd721c21c07855aa4ba37f5281d145`
- Verified live on 12 July 2026: hosted math returned `691`; hosted tool selection called `lookup_run_evidence` for G=2/G=16; offline verifier passed all eight checks.

`space_source/` is the exact local source snapshot. Hosted calls are labeled `LIVE HF ROUTER`; provider failures fall back to visibly labeled deterministic output.

## 2. Recorded fallback

- `hf_hosted_tool_math_captioned.mp4` — 44.52 seconds.
- `wandb_claims_captioned.mp4` — 38.12 seconds.

Both are H.264, 1600×900, 25 fps, with burned-in explanation captions.

## 3. Static fallback

Open `static_screenshots/index.html`. The seven frames are captured from the verified videos and ordered for a no-network walkthrough.

## 4. Offline executable fallback

Double-click `run_offline_dashboard.command`, or run:

```bash
./offline/run.sh
open offline/output/dashboard.html
```

The package verifies 14 artifact hashes, recomputes four 2,560-rollout Claim 2 arms, confirms the exact P4 contraction range `3.7627–12.1950%`, reconciles 983 broad Tinker objects with the 70+ curated telemetry corpus, and keeps the 19 gold rows as a separate subset.

## Presentation order

Use `PESU_MTech_Phase1_Session1_Review_ArvindCR_complete_demo_pack.pptx` when present in this folder.

1. Try the live Space.
2. If it is cold, play the embedded/captioned MP4.
3. If video playback fails, open the static walkthrough.
4. If all network/browser surfaces fail, run the offline command and show the PASS report.

Defense-safe wording: say “roughly 4–12% completion-length contraction”; the exact audit is 3.7627–12.1950%.
