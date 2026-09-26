# E9 small-scale (MLE-bench, original public subset)

Base Qwen3.6-35B-A3B on Tinker (non-thinking, temp 0, max_tokens 4096) acts as a few-step solution writer on 5 MLE-bench
low-complexity competitions. Scoring uses the official mlebench `grade_csv`.

Result: **any-medal 1/5 = 0.20** (Wilson95 0.036-0.624). Above-median is also 1/5, and 4/5 submissions were valid.
- nomad2018: silver. spooky, random-acts-of-pizza, leaf: valid but below median. detecting-insults: invalid (Comment column missing).

What ran:
- Pool: 6 low-split competitions with small tabular/text data. Five were drawn by `random.Random(20260926).sample`.
- `code/run_mle.py`: the prompt holds description.md, the file list and CSV heads. The actor returns one Python script, which is
  uploaded to the Colab CPU session `e9-mle` and run with a 20-min cap. If it crashes or writes no submission.csv, the
  error tail goes back to the actor, up to 3 attempts. Scripts, logs and submissions are in `raw/<comp>/`.
- `code/grade_all.py` writes `raw/grades.json`. Actor usage is in `raw/shim_usage.jsonl`, logged by `../E6/code/tinker_shim.py` on port 18770.

Rerun: `mlebench prepare -c <comp> --data-dir D` (the Kaggle creds in ~/.kaggle are already accepted), then `colab new -s e9-mle`,
start the shim, `run_mle.py D E9/raw <comps>`, `colab stop -s e9-mle`, and `grade_all.py D E9/raw <comps>` from a venv
with mle-bench-source installed.
Caveats: n=5, CPU only, minimal scaffold. This is a new arm and is not comparable to the earlier 40/75 campaign.
