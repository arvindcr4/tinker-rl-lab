# ai-scientist-template/ — INDEX

**Purpose:** Sakana AI-Scientist (v1) template for autonomous GRPO math-reasoning research: generates ideas, edits `experiment.py`, runs GSM8K, writes/reviews a LaTeX paper.

**Key files:**
- `README.md` — setup + how AI Scientist uses this template (cost, workflow).
- `experiment.py` — GRPO training on GSM8K via TRL GRPOTrainer + LoRA on small Qwen; the file the agent mutates.
- `prompt.json` — system + task_description fed to the AI Scientist (GRPO PhD-student persona, prior findings).
- `seed_ideas.json` — starter research ideas (partial-credit reward, curriculum, group-size/KL, etc.).
- `plot.py` — plotting of run metrics.
- `WRITEUP.md` — paper-writing notes / prior Tinker-scale results.
- `prompt.json`, `requirements.txt` — task framing + deps.
- `partial_credit_reward_shaping.pdf` — example generated paper output.

**Subfolders:**
- `latex/` — ICLR-style paper template (see its INDEX.md).
- `run_0/` — baseline experiment result JSON (see its INDEX.md).

**Find it fast:**
- to change the GRPO experiment → `experiment.py`
- to see/add seed ideas → `seed_ideas.json`
- to change agent framing → `prompt.json`
