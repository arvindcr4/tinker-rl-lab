# Vibe smells captured in deep-dive/ — fix ledger

Date: 2026-08-02

## A. Documentation smells (deep-dive artifacts themselves)

| Smell | Capture | Fix |
|---|---|---|
| P1–P12 audits were identical templates (only title/path differed) | `P*_antivibe_audit.md` | **Fixed** — rewritten as real senior audits grounded in portfolio verification |
| Minute-grain timestamps dirtied ~1344 per-file docs every run | `tools/apply_antivibe.py` | **Fixed** — date-grain + `SOURCE_DATE_EPOCH` |
| Paper-scoped deep-dives outside AntiVibe code scope still claimed ZVF theory for every paper | template generator | **Fixed** via rewrite above |

## B. Code smells named in subsystem dives

| Smell | Capture | Status |
|---|---|---|
| ~15 matrix cells silently ran the wrong framework | `framework-backend-dispatch-2026-08-02.md` | **Fixed** in tree: per-fw Modal `_PER_FW`, vast `--framework`, gcp unified dispatch, `test_each_cell_threads_its_framework` (37/37 pass) |
| Colab infinite recursion (`run_canonical` ↔ colab backend) | same | **Fixed**: `ColabBackend.run` → `dispatch_framework()` + test |
| verl PYTHONPATH shadowing repo root | `02-framework-integrations` | **Fixed**: `cwd=output_dir`, `PYTHONPATH=""` in `verl/trainer.py` |
| openrlhf + 27 experiment/atropos files stuffed with unactionable `TODO: Address …` review notes | deep-dive framework notes + repo-wide paste | **Fixed**: converted to `LIMITATION` prose across openrlhf/skyrl/tinker/hybrid (not fake work tickets) |
| Driver-exists tests green while matrix wrong | `framework-backend-dispatch` | **Fixed**: invariant tests check framework threading |

## C. Portfolio / paper integrity smells (evidence, not style)

Tracked in `drafts/PORTFOLIO_DECISION.md` and demoted in `PORTFOLIO_ROSTER_DISPOSITION.md`. Spine is P11; P1/P2 short notes ship; P3–P10/P12 not submission queue.

## D. Explicitly not fixed here (need GPU or human)

- Multi-seed openrlhf / held-out campaigns
- Confirmatory matrix / A004 bind
- Portal upload

## Regeneration

```bash
python3 tools/apply_antivibe.py          # per-file compact docs
# paper audits: edit deep-dive/P*_antivibe_audit.md (this file set) — do not template-clone
```
