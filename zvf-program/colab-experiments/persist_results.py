"""Persist Colab experiment results to the repo and to Weights & Biases.

Parses the saved run log (stdout from `colab run`), extracts the E1/E3 RESULT
JSON lines, writes them as files in results/, and logs each experiment as a W&B
run (auth via ~/.netrc). E2 failed on the VM (torchao conflict) -> status note.

Usage:  .venv/bin/python zvf-program/colab-experiments/persist_results.py
"""
import json, re, pathlib, datetime

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
LOG = OUT / "colab_run_b58dzsrjs.log"
WANDB_PROJECT = "zvf-colab-experiments"

text = LOG.read_text()
results = {}
for tag, fname in [("E1_RESULT", "e1_grad_signal.json"), ("E3_RESULT", "e3_open_audit.json")]:
    m = re.search(rf"^{tag} (\{{.*\}})\s*$", text, re.M)
    if not m:
        print(f"[persist] {tag}: NOT FOUND in log")
        continue
    obj = json.loads(m.group(1))
    obj["_source"] = "colab run --gpu T4 (job b58dzsrjs)"
    (OUT / fname).write_text(json.dumps(obj, indent=2) + "\n")
    results[tag] = obj
    print(f"[persist] wrote results/{fname}")

(OUT / "e2_lora_vs_fullft.STATUS.txt").write_text(
    "E2 FAILED on the Colab VM: peft pulled an incompatible torchao 0.10.0 "
    "(needs >0.16). Rerun after `pip install -U 'torchao>=0.16'` or removing the "
    "peft->torchao import path.\n")
print("[persist] wrote results/e2_lora_vs_fullft.STATUS.txt")

readme = f"""# Colab experiment results

Persisted from `colab run` stdout (job b58dzsrjs). Raw log: `colab_run_b58dzsrjs.log`.

| Exp | Status | Headline |
|-----|--------|----------|
| E1 grad-signal | done | corr(grad_norm, p(1-p)) = {results.get('E1_RESULT',{}).get('pearson_gradnorm_vs_p1mp','?')} (validates Theory T3) |
| E2 LoRA vs full-FT | FAILED | torchao 0.10.0 incompatible (needs >0.16); rerun pending |
| E3 open audit | done | DAPO drove ZVF->0 (+45% rollouts); adaptive-G + Dr.GRPO best held-out |

Toy 0.5B model on synthetic arithmetic -- directional evidence, not publishable effect sizes.
Also logged to W&B project `{WANDB_PROJECT}`.
"""
(OUT / "README.md").write_text(readme)
print("[persist] wrote results/README.md")

# ---- Weights & Biases ----
try:
    import wandb
except Exception as e:
    print(f"[wandb] not available ({e}); repo files written, skipping W&B.")
    raise SystemExit(0)

for tag, name in [("E1_RESULT", "E1_grad_signal"), ("E3_RESULT", "E3_open_audit")]:
    if tag not in results:
        continue
    obj = results[tag]
    run = wandb.init(project=WANDB_PROJECT, name=name, reinit=True,
                     config={k: v for k, v in obj.items()
                             if isinstance(v, (int, float, str, list))},
                     tags=["colab", "zvf", "toy-0.5B"])
    flat = {k: v for k, v in obj.items() if isinstance(v, (int, float))}
    wandb.summary.update(flat)
    if tag == "E1_RESULT":
        t = wandb.Table(columns=["difficulty", "mean_p", "mean_ZVF", "mean_grad_norm"])
        for d, v in obj["by_difficulty"].items():
            t.add_data(d, v["mean_p"], v["mean_ZVF"], v["mean_grad_norm"])
        wandb.log({"by_difficulty": t})
    if tag == "E3_RESULT":
        t = wandb.Table(columns=["arm", "mean_heldout_delta", "mean_zvf", "mean_rollouts"])
        for arm, v in obj["by_arm"].items():
            t.add_data(arm, v["mean_heldout_delta"], v["mean_zvf"], v["mean_rollouts"])
        wandb.log({"by_arm": t})
    print(f"[wandb] logged {name}: {run.url}")
    wandb.finish()
print("[persist] done.")
