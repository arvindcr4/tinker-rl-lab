# External platform audit — Lightning AI, Google Colab, Modal

Companion to `RECTIFICATION.md` (Tinker × W&B × HF). Audited 2026-07-11.

## Lightning AI — checked, EMPTY (clean negative)

Authenticated as `arvindcr4` via lightning-sdk (credentials persisted by the
SDK login; key was pasted in chat — rotate it). Full enumeration:

- Teamspaces: 1 (`playground`)
- Studios: 5 — `openclaw`, `scratch-studio`, `personal-gpt`, `litdata`,
  `talk-to-gpt` (none RL/tinker-related)
- **Jobs: 0**

Verdict: no experiment provenance exists on Lightning. No paper claim should
cite it; conversely no runs are lost there.

## Google Colab — notebooks live in Drive; run provenance is W&B-only

There is no Colab API that lists executed sessions; Drive holds the
notebooks. Drive enumeration (owner = arvindcr4@gmail.com, mimeType
`application/vnd.google.colaboratory`, ~50+ notebooks) found exactly two
relevant to this program:

- `ppo_reinforce_baselines_colab.ipynb` (created 2026-04-04, 326 KB) —
  drive id `1vZprcGaj_FT02wTYC320dCMdHyDxUZ6d`. Almost certainly the
  classic-control PPO/REINFORCE baseline notebook behind the cross-framework
  comparison (F3 / P5 supporting evidence).
- `training-tinker.ipynb` (created 2026-01-12, 17 KB) — drive id
  `1kWpbcldbWB56ZuSbvMyo8FnrKc0hy4rj`. Early Tinker training experiment.

Everything else Colab-related in Drive is unrelated (OCR, unsloth, legal
fine-tuning, coursework) or an `Untitled*.ipynb`.

Consequences for the claim table:
- The 16-run `zvf-colab-experiments` W&B project remains the ONLY run-level
  provenance for the Colab arms (P5-C2, P7-C2, P7-C3). The e1–e5 sources in
  `zvf-program/colab-experiments/*.py` are the code record; no additional
  run identity is recoverable from Drive.
- `ppo_reinforce_baselines_colab.ipynb` should be linked from the P5/F3
  materials as the baseline notebook artifact (drive id above).

## Modal — four volumes of open-stack artifacts (see MODAL_INVENTORY.md)

Auth pre-existing in `~/.modal.toml` (token id matches the one provided).
One deployed app (`claude-science-tulip-mist-shell`, not training). Four
volumes hold the open-stack experiment outputs:

- `tinkerrl-zvf-open-results` — ZVF open audit, Qwen2.5-0.5B, G ∈ {4,8,32}
- `tinker-results` — cross-framework zoo (trl_grpo/sft/dpo/ppo, sb3,
  cleanrl, tianshou, pufferlib, rl_games, d3rlpy + summary.json; 5 seeds
  42/123/456/789/1024; d3rlpy 0/5 successful per its own summary)
- `tinkerrl-results` — `drgrpo_gsm8k` (grpo & dr_grpo × s42/s123/s456 —
  the P4 `local:` runs), `samestack` (grpo & ppo × 5 seeds — P5-C3),
  `groupsize_zvf`, `llama_heldout`, math arms
- `tinker-rl-results` — empty

These `modal://volume/path` refs upgrade the claim table's bare `local:` IDs
to resolvable remote artifacts. Inventory + downloaded metrics JSONs:
`../modal_registry/` (generator `modal_inventory.py`).

## Key rotation reminder (all pasted in chat)

- W&B `wandb_v1_…` — ALSO was hardcoded in git history (see commit c0edf76)
- HF `hf_…`
- Tinker `tml-…` (also in repo-root `.env`, gitignored)
- Lightning `LIGHTNING_API_KEY`
- Modal token id was already present in `~/.modal.toml` (no new exposure)
