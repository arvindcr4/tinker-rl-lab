# TinkerRL Measurement Lab (web demo)

Interactive defense demo built with React. It has nine views: an overview; a GRPO advantage playground; zero-variance groups against the binomial identity; the same-stack GRPO-vs-PPO results, including the v1 → v2 correction; the cross-platform calibration (Modal A10G vs Colab A100); real stored model answers, re-graded live with a grader you can probe; a `stackdiff` explorer over the registry; the thesis number audit; and the hard viva questions, each linked to its evidence. Every number is computed from stored results. Nothing trains, and nothing calls the network.

## Open it

Open `tinkerrl_demo.html` in a browser. It is one self-contained file and works offline from disk. Fonts fall back to system faces without internet.

Presenting: ← / → or 1–9 move through the views in demo order. P (or the Present button) switches to larger text for a projector. Suggested 90-second route:
1. Advantage lab: tap "All correct" to show a zero-variance group.
2. GRPO vs PPO: toggle v1 then v2 to show the correction.
3. Platform calibration: the base accuracy per GPU, then the platform effect against the ±0.02 band.
4. Real answers: open "only GRPO fixed it" and read one item.
5. Stack diff: the Tinker vs OpenRLHF "grpo" preset.

## Rebuild after new results

From this directory:

```sh
npm install
npm run data    # python3 build_demo_web.py -> src/data.json
npm run build   # vite -> dist/index.html, copied to tinkerrl_demo.html
npm run dev     # live development server
npm test        # outside checks, below
```

`npm run data` reads `platform_hybrid/experiments/results/samestack_gsm8k_v2s/*.json`, the stack registry (`platform_hybrid/registry/query.py`), `reports/public_revision_2026-10-11/PUBLIC_CHECKOUT_NUMBER_AUDIT.json` and, if present, `reports/project_finish_2026-10-11/validation/samestack_v2/platform_calibration.json`. When the remaining PPO seeds and the `ppo_8x8` arm land, rerun both commands. The provisional labels then clear by themselves.

The paired t interval, TOST and exact sign-flip p in `src/stats.js` match `reports/project_finish_2026-10-11/validation/samestack_v2/v2_analysis.json` to float precision.

## Checks

`npm test` runs two checks that do not trust the page:

- `tests/grade_all.mjs` re-grades every stored completion (71,226 answers from 27 runs) with the in-page grader and requires agreement with the stored scores.
- `tests/test_demo.py` drives the built page in Chrome. It checks that every view renders at desktop and phone width in both themes, with no overflow or console errors. It also checks that the in-page statistics match `v2_analysis.json`, that the grader probe follows the strict rule, and that the presenter keys work.
