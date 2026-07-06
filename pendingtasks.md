# Pending Tasks

_Last updated: 2026-07-06. Owner tags: **[you]** = needs Arvind's input/action · **[me]** = I can do autonomously._

## 🔴 P0 — ESA Phase-1 (graded, due 11/12 July)

| # | Task | Owner | Next action |
|---|------|-------|-------------|
| 1 | **Deck: Review-3 panel remarks** (slide 5) | **[you]** | Send the panel's *actual* wording from the 4/5 July call — I've inserted a plausible draft that must be replaced/confirmed. |
| 2 | **Deck: completion %** (slide 13) | **[you]** | Confirm/adjust the `~65%` I set. |
| 3 | **Integrate the 8 TikZ diagrams into the report** | [me] | `\usepackage{tikz,pgfplots}` + `\input` each figure at its section; recompile. Figures ready in `reports/esa_phase1/tikz/diagrams.tex`. |
| 4 | **Report: confirm AERO/AVSPO citation labels** | **[you]**/[me] | arXiv IDs verified, but the *acronym→paper* mapping for AERO (2602.14338) and AVSPO (2605.21125) needs a human eyeball — confirm these are the papers you mean. |
| 5 | **Hard-copy per "MTech Project Hard Copy Guidelines"** | [me]+**[you]** | Format the report to the binding/margin/title-page spec (Olympus module 8695029), then print. |
| 6 | **Rehearse the code walkthrough / demo** | **[you]** | Runbook is `reports/esa_phase1/CODE_WALKTHROUGH.md`; dry-run Demos A/B/C once before the panel. |
| 7 | **Final proofread of report + deck** | **[you]** | Read `ESA_Phase1_Report_DRAFT.pdf` end-to-end; check the honest-negatives framing reads as intended. |

## 🟡 P1 — Research / Phase-2 (toward the 8 papers / ICLR 2027)

| # | Task | Owner | Notes |
|---|------|-------|-------|
| 8 | **P1: scaled layer-freeze run** | [me] | Bigger model + GSM8K + multi-seed on Colab L4 (`colab run`), to turn the 18–39% freeze-fraction *estimate* into a measured FLOP saving. Strongest positive result — worth hardening. |
| 9 | **P2/P3: token-budget-optimal curriculum** | [me] | The *better* lever (naive filtering already shown not to beat baseline). Design multi-seed from the start with staleness bounds. |
| 10 | **P4: length-bias / KL-surprise mask experiment** | [me] | White-box (Colab); currently only designed. |
| 11 | **P5: sign provenance records + CI gate** | [me] | ed25519-sign `*.provenance.json`; wire `minreport.py verify --strict` so a run can't publish below grade B. |
| 12 | **P7: ZVF controller (PID) implementation** | [me] | Currently designed only; needs a live control loop. |
| 13 | **P8: proper eval** | [me] | More seeds, record base model/task, ship a reproducible analysis script (currently prose-only), fix the 3:1 class imbalance note. |
| 14 | **Write the 8 papers (P1–P8)** | [me]+**[you]** | Drafts in `paper/`; P5 (flagship) + P1 (technical) are the lead bets. |

## 🟢 P2 — Infra & hygiene

| # | Task | Owner | Notes |
|---|------|-------|-------|
| 15 | **Rotate the W&B API key** | **[you]** | It was pasted in chat earlier; rotate at wandb.ai/settings and update `.env.minimax`. |
| 16 | **Commit all new work to git** | [me] | New: `experiments/openings/{campaign,parallel_sweep,p2_collapse_analysis}.py`, `registry/provenance/*`, `reports/esa_phase1/*` (report, walkthrough, diagrams, verification). Currently uncommitted. |
| 17 | **Fix `smoke_test.sh` python path** | [me] | Uses bare `python`; point at `.venv/bin/python` so the offline demo fallback is reliable. |
| 18 | **Campaign metrics bug — already fixed** | ✅ | `campaign.py` loss-fn now returns `{"loss": ...}`; prior batch's `zero_loss_frac` invalid (noted in findings). |

## ✅ Done this session (for reference)
- Consolidated MiniMax P5–P8 into main; adversarial review (Gemini/GPT-5.5 Pro/kimi/codex).
- Experiments: P2 (ZVF 72–77%), P3 (no G4 sweet spot), curriculum (null, multi-seed), **P1 white-box (predictable, overlap=1.0)**, P8 (AUROC 0.63).
- Parallel Tinker runner + verified Colab L4 white-box path.
- **P5 flagship built + working** (`registry/provenance/minreport.py` — grades runs A–F).
- ESA report (11pp, 0 todos, verified citations), code walkthrough, deck (guide filled), 8 TikZ diagrams.
