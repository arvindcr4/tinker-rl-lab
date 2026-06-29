"""Persist the E2 production run (Qwen3-4B-Instruct-2507, 3 seeds).

Reads the Colab run log for the E2_RESULT line emitted by
e2_lora_vs_fullft_4b.py, writes results/e2_lora_vs_fullft_4b.json,
refreshes the E2 production README, and logs to W&B project
zvf-colab-experiments (separate run from the 0.5B pilot).

Usage:
  .venv/bin/python zvf-program/colab-experiments/persist_e2_prod.py [LOG_FILE]

Defaults to results/e2_prod.log if LOG_FILE not given.
"""
import json, re, sys, pathlib
import datetime as _dt

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "results"
WANDB_PROJECT = "zvf-colab-experiments"

LOG_FILE = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else (OUT / "e2_prod.log")
OUT_JSON = OUT / "e2_lora_vs_fullft_4b.json"
README = OUT / "README_E2_PROD.md"


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
            f"(std {l['std_heldout_delta']:.3f}) vs full-FT {f['mean_heldout_delta']:+.3f} "
            f"(std {f['std_heldout_delta']:.3f}); LoRA-full gap {o['delta_lora_minus_full']:+.3f}; "
            f"mean ZVF: LoRA {l['mean_zvf']:.3f} vs full-FT {f['mean_zvf']:.3f}")


def main():
    obj = parse_log(LOG_FILE)
    if obj is None:
        sys.exit(1)

    obj["_source"] = f"colab run --gpu A100 ({LOG_FILE.name})"
    obj["_pillar"] = "P4"
    obj["_collected_utc"] = _dt.datetime.utcnow().isoformat() + "Z"
    OUT_JSON.write_text(json.dumps(obj, indent=2) + "\n")
    print(f"[persist] wrote {OUT_JSON}")

    readme = (
        "# E2 production run (Qwen3-4B-Instruct-2507, 3 seeds)\n\n"
        f"Logged to W&B `{WANDB_PROJECT}` (run name `E2_lora_vs_fullft_4b`).\n"
        f"Pillar 4. Held-out on {obj['heldout_n']} synthetic-arithmetic problems.\n"
        f"Per-seed trajectories in `e2_lora_vs_fullft_4b.json`.\n\n"
        f"**Headline:** {headline(obj)}\n\n"
        f"- Steps per arm: {obj['steps']}\n"
        f"- Group size G: {obj['group_size']}\n"
        f"- Batch: {obj['batch']}\n"
        f"- LoRA rank/alpha/dropout: 16/32/0\n"
        f"- LoRA targets: q_proj, k_proj, v_proj, o_proj\n"
        f"- LR: LoRA={obj['lr_lora']}, full-FT={obj['lr_full']}\n"
        f"- Heldout N: {obj['heldout_n']} (matched between arms; seed-reset each run)\n"
        f"- Seed reset per arm via `random.seed(s); torch.manual_seed(s)` so heldout set is reproducible\n\n"
        "## Caveat (honest scope)\n\n"
        "- Synthetic arithmetic, not GSM8K — directional evidence on the LoRA↔full axis.\n"
        "- Tinker side is LoRA-only; this script's full-FT arm is the comparison Tinker can't run.\n"
        "- 3 seeds is the minimum for std-dev reporting; consider 5+ for any production claim.\n"
        "- 4B is the smallest model where full-FT is memory-tight on A100 40GB; larger models\n"
        "  require LoRA-only or gradient-accumulation tricks.\n"
    )
    README.write_text(readme)
    print(f"[persist] wrote {README}")

    # W&B
    try:
        import wandb
    except Exception as e:
        print(f"[wandb] unavailable ({e}); repo files written, skipping W&B.")
        return

    run = wandb.init(
        project=WANDB_PROJECT,
        name="E2_lora_vs_fullft_4b",
        reinit=True,
        config={k: v for k, v in obj.items()
                if isinstance(v, (int, float, str, list)) and k != "trajectory"},
        tags=["colab", "zvf", "qwen3-4b-2507", "P4", "e2-prod"],
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
    wandb.summary.update(summary)
    # Per-seed trajectories
    for arm_name in ("lora", "full"):
        for r in obj[arm_name]["per_seed"]:
            wandb.log({f"{arm_name}/seed{r['seed']}/heldout_delta": r["heldout_delta"],
                       f"{arm_name}/seed{r['seed']}/mean_zvf": r["mean_zvf"]})
    print(f"[wandb] logged E2_lora_vs_fullft_4b: {run.url}")
    wandb.finish()


if __name__ == "__main__":
    main()