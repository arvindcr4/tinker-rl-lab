"""Persist the E2 production v2 run (5 seeds, harder task).

Reads the Colab run log for E2_RESULT from e2_lora_vs_fullft_4b_v2.py,
writes results/e2_lora_vs_fullft_4b_v2.json, refreshes README_E2_PROD_v2.md,
and logs to W&B project zvf-colab-experiments. Compares against v1 in the
README so reviewers can see the v1-vs-v2 contrast.

Usage:
  .venv/bin/python zvf-program/colab-experiments/persist_e2_prod_v2.py [LOG_FILE]

Defaults to results/e2_prod_v2.log if LOG_FILE not given.
"""
import json, re, sys, pathlib
import datetime as _dt

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "results"
WANDB_PROJECT = "zvf-colab-experiments"
V1_JSON = OUT / "e2_lora_vs_fullft_4b.json"

LOG_FILE = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else (OUT / "e2_prod_v2.log")
OUT_JSON = OUT / "e2_lora_vs_fullft_4b_v2.json"
README = OUT / "README_E2_PROD_v2.md"


def parse_log(path: pathlib.Path):
    if not path.exists():
        print(f"[persist] log {path} missing"); return None
    m = re.search(r"^E2_RESULT (\{.*\})\s*$", path.read_text(), re.M)
    if not m:
        print(f"[persist] E2_RESULT line not found in {path}"); return None
    return json.loads(m.group(1))


def headline(o):
    l, f = o["lora"], o["full"]
    n = l["n_seeds"]
    return (f"{n}-seed mean heldout_delta: LoRA {l['mean_heldout_delta']:+.3f} "
            f"(std {l['std_heldout_delta']:.3f}) vs full-FT "
            f"{f['mean_heldout_delta']:+.3f} (std {f['std_heldout_delta']:.3f}); "
            f"LoRA-full gap {o['delta_lora_minus_full']:+.3f}; "
            f"mean ZVF: LoRA {l['mean_zvf']:.3f} vs full-FT {f['mean_zvf']:.3f}")


def v1_summary():
    """Load v1 results for cross-reference, if present."""
    if not V1_JSON.exists():
        return None
    try:
        v1 = json.loads(V1_JSON.read_text())
        return {
            "task": "a+b in [11,60] (saturated)",
            "n_seeds": v1["lora"]["n_seeds"],
            "lora_mean_delta": v1["lora"]["mean_heldout_delta"],
            "full_mean_delta": v1["full"]["mean_heldout_delta"],
            "delta_lora_minus_full": v1["delta_lora_minus_full"],
        }
    except Exception:
        return None


def main():
    obj = parse_log(LOG_FILE)
    if obj is None:
        sys.exit(1)

    obj["_source"] = f"colab run --gpu A100 ({LOG_FILE.name})"
    obj["_pillar"] = "P4"
    obj["_collected_utc"] = _dt.datetime.now(_dt.timezone.utc).isoformat()
    OUT_JSON.write_text(json.dumps(obj, indent=2) + "\n")
    print(f"[persist] wrote {OUT_JSON}")

    v1 = v1_summary()
    comparison = ""
    if v1:
        comparison = (
            f"\n## Comparison with v1 (saturated task)\n\n"
            f"| Metric | v1 (a+b, saturated) | v2 (a*b, harder) |\n"
            f"|:-------|----------------------:|----------------:|\n"
            f"| Task | {v1['task']} | {obj['task']} |\n"
            f"| Seeds | {v1['n_seeds']} | {obj['lora']['n_seeds']} |\n"
            f"| LoRA mean delta | {v1['lora_mean_delta']:+.3f} | "
            f"{obj['lora']['mean_heldout_delta']:+.3f} |\n"
            f"| Full-FT mean delta | {v1['full_mean_delta']:+.3f} | "
            f"{obj['full']['mean_heldout_delta']:+.3f} |\n"
            f"| LoRA-full gap | {v1['delta_lora_minus_full']:+.3f} | "
            f"{obj['delta_lora_minus_full']:+.3f} |\n\n"
        )

    readme = (
        "# E2 production run v2 (Qwen3-4B-Instruct-2507, 5 seeds, harder task)\n\n"
        f"Logged to W&B `{WANDB_PROJECT}` (run name `E2_lora_vs_fullft_4b_v2`).\n"
        f"Pillar 4. Held-out on {obj['heldout_n']} 3-digit-by-2-digit multiplication\n"
        f"problems.\n\n"
        f"**Headline:** {headline(obj)}\n\n"
        f"- Steps per arm: {obj['steps']}\n"
        f"- Group size G: {obj['group_size']}\n"
        f"- Batch: {obj['batch']}\n"
        f"- LoRA rank/alpha/dropout: 16/32/0\n"
        f"- LoRA targets: {', '.join(obj['lora_targets'])}\n"
        f"- LR: LoRA={obj['lr_lora']}, full-FT={obj['lr_full']}\n"
        f"- Heldout N: {obj['heldout_n']} (matched between arms; seed-reset each run)\n"
        f"- Seed reset per arm via `random.seed(s); torch.manual_seed(s)` so heldout set is reproducible\n\n"
        "## Why a harder task\n\n"
        "The v1 run (a+b in [11,60]) saturated to ceiling in 1-3 steps because\n"
        "Qwen3-4B-Instruct-2507 trivially handles single-digit addition. The result\n"
        "defended the LoRA-only Tinker constraint but did not test whether LoRA\n"
        "wins under non-saturated conditions. This v2 run uses 3-digit-by-2-digit\n"
        "multiplication (range [100,999] x [11,99], answers [1100, 98901]) where the\n"
        "model cannot solve perfectly in 80 steps, so within-group variance stays\n"
        "live throughout training and the gradient signal is meaningful.\n\n"
        f"{comparison}"
        "## Caveats (honest scope)\n\n"
        "- Synthetic arithmetic, not GSM8K — directional evidence on the LoRA vs full-FT axis.\n"
        "- 5 seeds (the floor for std-dev reporting in deep-RL best practice).\n"
        "- 4B is the smallest model where full-FT is memory-tight on A100 40GB; the\n"
        "  scaled-up LoRA targets (q/k/v/o/gate/up/down) match QLoRA paper recipe.\n"
        "- Task difficulty chosen to keep ZVF live throughout training. Results\n"
        "  generalize across tasks only when the difficulty is similar.\n"
    )
    README.write_text(readme)
    print(f"[persist] wrote {README}")

    try:
        import wandb
    except Exception as e:
        print(f"[wandb] unavailable ({e}); repo files written, skipping W&B.")
        return

    run = wandb.init(
        project=WANDB_PROJECT,
        name="E2_lora_vs_fullft_4b_v2",
        reinit=True,
        config={k: v for k, v in obj.items()
                if isinstance(v, (int, float, str, list)) and k != "trajectory"},
        tags=["colab", "zvf", "qwen3-4b-2507", "P4", "e2-prod-v2", "harder-task"],
    )
    summary = {
        "lora_mean_heldout_delta": obj["lora"]["mean_heldout_delta"],
        "lora_std_heldout_delta": obj["lora"]["std_heldout_delta"],
        "lora_mean_zvf": obj["lora"]["mean_zvf"],
        "full_mean_heldout_delta": obj["full"]["mean_heldout_delta"],
        "full_std_heldout_delta": obj["full"]["std_heldout_delta"],
        "full_mean_zvf": obj["full"]["mean_zvf"],
        "delta_lora_minus_full": obj["delta_lora_minus_full"],
        "n_seeds": obj["lora"]["n_seeds"],
        "headline": headline(obj),
    }
    if v1:
        summary["v1_delta_lora_minus_full"] = v1["delta_lora_minus_full"]
    wandb.summary.update(summary)
    for arm_name in ("lora", "full"):
        for r in obj[arm_name]["per_seed"]:
            wandb.log({f"{arm_name}/seed{r['seed']}/heldout_delta": r["heldout_delta"],
                       f"{arm_name}/seed{r['seed']}/mean_zvf": r["mean_zvf"]})
    print(f"[wandb] logged E2_lora_vs_fullft_4b_v2: {run.url}")
    wandb.finish()


if __name__ == "__main__":
    main()