# P6 shared-file requests (coordinator)

1. `platform_hybrid/paper/references.bib`: see `P6_amp_requests.md` §R7. Fix MO-GRPO to arXiv:2509.22047 (Y. Ichihara) and GRPO-LEAD to J. Zhang et al. Add AERO, arXiv:2602.14338. Make the minor Limozin, Liu "Tricks or Traps", and Le et al. fixes.
2. `platform_hybrid/registry/entries/` (used by P5 and P6): move the 11 `zvf130_*.json` records to `registry/excluded_unsourced/`. They were built from a simulation-projection risk-index fixture (rule 2). Then:
   - `delta_aero.json`: change the citation `le2025rlzvp` → AERO arXiv:2602.14338.
   - `tinker_aero`: change the status `implemented` → `absent`, because G stayed fixed at 8 in N2.
   - `tinker_areal` and `tinker_gift`: change the status → `surrogate`.
   - Family C `ci_method.source`: currently it points at the N2 bootstrap; correct it.
   - `delta_mcgrpo` 'mcts_rollout' and `delta_cppo` 'continuity_penalty': both mechanism names are wrong (finding 370).
   - `registry/query.py antiherding` and the schema `zvf_antiherding` / `joint_controller_predictions` blocks still expose the withdrawn δ_div construct (finding 351).
