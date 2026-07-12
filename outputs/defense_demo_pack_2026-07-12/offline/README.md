# Offline Defense Fallback

This folder is the no-network fallback for the TinkerRL MTech defense. It is self-contained: the audited JSONs, run-audit workbook, synthetic fixture, and exported W&B figures are copied under `data/` and `assets/`, with SHA-256 values pinned in `data/manifest.json`.

## 30-second start

From the repository root:

```bash
./submission/demo/defense_fallback/run.sh
```

Expected first line:

```text
OFFLINE DEFENSE CHECKS: PASS
```

Then open the generated dashboard:

```bash
open submission/demo/defense_fallback/output/dashboard.html
```

Or serve it on localhost, still without internet:

```bash
./submission/demo/defense_fallback/run.sh --serve
# http://127.0.0.1:8771/output/dashboard.html
```

## What the CLI proves

1. **Byte provenance:** all 14 copied evidence inputs match the pinned SHA-256 manifest.
2. **Claim 2 matched budget:** four corrected E-R2b JSONs each represent exactly 2,560 rollouts. The two G=2 arms have last-10 reward means 0.9000/0.9625 and ZVF means 0.975/0.975; the two G=16 arms have reward means 0.321875/0.3890625 and ZVF means 0.150/0.100.
3. **P4 length behavior:** using the audited first-5 to last-10 definition, all six corrected arms contract by 3.7627–12.1950%, which is the defense-safe “approximately 3.8–12.2%” wording.
4. **983 versus 70+:** the workbook is parsed with the Python standard library. It checks 983 rows in `runs`, 19 rows in `key_runs`, and the reconciliation text in `insights`. The totals use different inclusion rules: broad Tinker client objects versus the curated cross-library telemetry corpus.
5. **ZVF arithmetic:** the existing didactic fixture recomputes ZVF=0.5 and gradient utilization=0.5.

These checks establish local byte integrity and arithmetic consistency. They do not re-run training, establish causality, or turn a two-seed panel into a universal group-size law.

## Notebook walkthrough

Open `offline_defense_demo.ipynb` and choose **Run All**. It uses only Python’s standard library and package-relative files. The required narrative order is Goal → Setup → Steps → Checks → Next Steps.

To re-execute from the command line:

```bash
python3 -m jupyter nbconvert --execute --to notebook --inplace \
  submission/demo/defense_fallback/offline_defense_demo.ipynb
```

## Defense order and fallback ladder

1. Run `run.sh` and show the PASS summary.
2. Open `output/dashboard.html`; point to the exact Claim 2 and P4 tables.
3. Show the two exported W&B figures beneath the recomputed tables.
4. If the browser fails, open `output/report.json` or run `run.sh --json`.
5. If Jupyter fails, the CLI remains the authoritative zero-dependency path.

Suggested explanation: “These are frozen local exports covered by SHA-256. The script reads the raw result JSONs and the XLSX XML directly, recomputes the numbers, and fails closed if any byte or expected value changes.”

## Files

```text
defense_fallback/
├── run.sh                    # one-command entry point
├── run_checks.py             # stdlib-only checks and report generator
├── offline_defense_demo.ipynb
├── dashboard_template.html
├── README.md
├── data/                     # fixture, workbook, raw result JSONs, manifest
├── assets/                   # copied W&B exports
└── output/                   # generated report.json and dashboard.html
```
