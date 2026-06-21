"""Persist the E2-rerun + E4-E7 Colab batch to the repo and to W&B.

Parses each run log for its `E{n}_RESULT {json}` line, writes results/e{n}_*.json,
refreshes the results README with the new batch, and logs each experiment to W&B
project `zvf-colab-experiments`. Idempotent; skips any experiment whose RESULT
line is absent.

Usage:  .venv/bin/python zvf-program/colab-experiments/persist_e2_e7.py
"""
import json, re, pathlib

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "results"
WANDB_PROJECT = "zvf-colab-experiments"

# (RESULT tag, source log, output json, W&B run name, pillar)
SPECS = [
    ("E2_RESULT", "e2_rerun.log", "e2_lora_vs_fullft.json", "E2_lora_vs_fullft", "P4"),
    ("E4_RESULT", "e4_run.log",   "e4_scaling_law.json",    "E4_scaling_law",    "P1"),
    ("E5_RESULT", "e5_run.log",   "e5_grad_geometry.json",  "E5_grad_geometry",  "P2"),
    ("E6_RESULT", "e6_run.log",   "e6_live_triage.json",    "E6_live_triage",    "P3"),
    ("E7_RESULT", "e7_run.log",   "e7_stack_levers.json",   "E7_stack_levers",   "P4"),
]

def parse_log(tag, logname):
    f = OUT / logname
    if not f.exists():
        print(f"[persist] {tag}: log {logname} missing"); return None
    m = re.search(rf"^{tag} (\{{.*\}})\s*$", f.read_text(), re.M)
    if not m:
        print(f"[persist] {tag}: RESULT line not found in {logname}"); return None
    return json.loads(m.group(1))

results = {}
for tag, logname, outname, _, pillar in SPECS:
    obj = parse_log(tag, logname)
    if obj is None:
        continue
    obj["_source"] = f"colab run --gpu T4 ({logname})"
    obj["_pillar"] = pillar
    (OUT / outname).write_text(json.dumps(obj, indent=2) + "\n")
    results[tag] = obj
    print(f"[persist] wrote results/{outname}")

def headline(tag, o):
    if tag == "E2_RESULT":
        a = {x["mode"]: x for x in o["arms"]}
        return f"LoRA Δ={a['lora']['heldout_delta']:+.2f} vs full-FT Δ={a['full']['heldout_delta']:+.2f} held-out"
    if tag == "E4_RESULT":
        return f"ZVF(p,K)=p^K+(1-p)^K fits R²={o['closed_form_r2']}; fp32 moves ZVF by {o['precision_side_check_K8']['delta_zvf_fp32_minus_bf16']:+.3f}"
    if tag == "E5_RESULT":
        return f"signal↔p(1-p) r={o['corr_signal_p1mp']} > signal↔GU r={o['corr_signal_gu']}; Fisher↔p(1-p) r={o['corr_fisher_p1mp']}"
    if tag == "E6_RESULT":
        a = o["by_arm"]
        return ("fixed Δ={:.2f}/ZVF={:.2f} | adaptiveG Δ={:.2f}/ZVF={:.2f} | +drop Δ={:.2f}/ZVF={:.2f} (matched ~{} rollouts)"
                .format(a['fixed_G']['mean_heldout_delta'], a['fixed_G']['mean_zvf'],
                        a['adaptiveG']['mean_heldout_delta'], a['adaptiveG']['mean_zvf'],
                        a['adaptiveG_drop']['mean_heldout_delta'], a['adaptiveG_drop']['mean_zvf'],
                        o['budget_rollouts']))
    if tag == "E7_RESULT":
        bl = o["by_lever"]
        parts = [f"{k}: ΔZVF={v['delta_zvf_vs_ref'][0]:+.2f}±{v['delta_zvf_vs_ref'][1]:.2f}"
                 for k, v in bl.items() if k != "reference"]
        return "stack-lever ΔZVF vs ref — " + "; ".join(parts)
    return ""

# ---- README ----
rows = []
for tag, _, _, name, pillar in SPECS:
    if tag in results:
        rows.append(f"| {name.split('_')[0]} | {pillar} | done | {headline(tag, results[tag])} |")
readme = ("# Colab experiment results — E2 rerun + E4–E7 batch\n\n"
          "New Colab-only batch (each requires a capability closed/LoRA-only/fixed-stack "
          "**Tinker** lacks), one per ZVF-Program pillar. Codex (gpt-5.5) reviewed the plan "
          "before launch; fixes in `PLAN_E4_E7.md`. Toy 0.5B on synthetic arithmetic — "
          "directional evidence, not publishable effect sizes. Logged to W&B `"
          + WANDB_PROJECT + "`.\n\n"
          "| Exp | Pillar | Status | Headline |\n|-----|--------|--------|----------|\n"
          + "\n".join(rows) + "\n")
(OUT / "README_E4_E7.md").write_text(readme)
print("[persist] wrote results/README_E4_E7.md")

# ---- W&B ----
try:
    import wandb
except Exception as e:
    print(f"[wandb] unavailable ({e}); repo files written, skipping W&B."); raise SystemExit(0)

for tag, _, _, name, pillar in SPECS:
    if tag not in results:
        continue
    obj = results[tag]
    run = wandb.init(project=WANDB_PROJECT, name=name, reinit=True,
                     config={k: v for k, v in obj.items() if isinstance(v, (int, float, str))},
                     tags=["colab", "zvf", "toy-0.5B", pillar, "E4-E7-batch"])
    wandb.summary.update({k: v for k, v in obj.items() if isinstance(v, (int, float))})
    wandb.summary["headline"] = headline(tag, obj)
    print(f"[wandb] logged {name}: {run.url}")
    wandb.finish()
print("[persist] done.")
