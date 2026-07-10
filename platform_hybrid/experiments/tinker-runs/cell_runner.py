#!/usr/bin/env python3
"""
cell_runner.py — THIN per-cell shim for the ZVF Pillar-1/M1 sweep.

WHAT THIS IS
  The sweep orchestrator (zvf-program/sweep/run_sweep.py) shells out once per
  run-cell to this script with a fixed set of flags:

      python3 cell_runner.py \
          --tag <t> --model <m> --task <task> \
          --loss <importance_sampling|grpo|dapo|gspo|drgrpo> \
          --seed <s> --group-size <G> --lr <lr> --steps <n> --rank <r> \
          --difficulty <easy|medium|hard> --out <path.json>

  This shim does NOT reimplement training. It maps those flags onto the
  arguments of the EXISTING, in-repo Tinker GRPO runner (live_zvf_probe.run)
  and calls it. That runner is the only piece that touches Tinker / does real
  compute, and it is the one that already logs per-step ZVF (zero-variance
  fraction) and GU (gradient utilisation = 1 - ZVF) into `step_log`.

WHAT THIS NEVER DOES
  * It never fabricates a metric, reward, ZVF, GU, loss, or "win". Every number
    in the output JSON comes from the real runner's own `result` dict.
  * If TINKER_API_KEY / credentials are missing it FAILS LOUDLY and writes a
    failure stub (status="failed"); it does NOT emit a fake completed result.

OUTPUT
  * Writes the runner's rich result JSON (master_results columns + reward_trace
    + per-step step_log with zvf/gu) to --out.
  * Idempotently upserts a one-line row into experiments/master_results.csv so
    the sweep's resume logic and aggregate_results.py see the cell. (The
    canonical regenerator remains experiments/aggregate_results.py, which
    rebuilds the CSV from the per-run JSONs; this upsert is a convenience.)

MAPPING ASSUMPTIONS (please confirm — see README at bottom of run_sweep docstring)
  --difficulty -> GSM8K reasoning-step terciles. GSM8K ships no difficulty
                  labels, so we bucket each training example by the number of
                  reasoning steps in its reference solution (count of '\n' /
                  '<<...>>' calculator annotations): easy=fewest steps tercile,
                  medium=middle, hard=most. The bucket only filters which
                  prompts the existing runner trains on; it changes no training
                  logic.
  --loss       -> the existing runner uses one GRPO surrogate (REINFORCE with a
                  per-group standardized advantage == the canonical
                  "importance_sampling" GRPO surrogate already in the repo).
                  We record the requested arm faithfully and apply the
                  arm-specific advantage post-processing the loop supports:
                    importance_sampling / grpo : per-group standardized adv (baseline)
                    dapo                       : grpo + asymmetric (DAPO-style)
                                                 advantage clip, no length bias
                    gspo                       : grpo with sequence-level (mean,
                                                 not sum) log-prob aggregation
                  These are surrogate approximations of the named losses on top
                  of the in-repo Tinker loop, NOT full re-implementations of
                  TRL/DAPO/GSPO. The arm is always recorded in the output so the
                  audit can stratify by it.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent                 # .../experiments/tinker-runs
REPO_ROOT = HERE.parents[1]                             # .../tinker-rl-lab
MASTER_CSV = REPO_ROOT / "experiments" / "master_results.csv"

# Column order MUST match the existing experiments/master_results.csv header so
# aggregate_results.py and run_sweep.py's resume logic keep working.
MASTER_COLS = [
    "group", "experiment_id", "experiment_name", "model", "model_short",
    "method", "task", "platform", "seed", "steps_completed", "peak_reward",
    "last10_avg", "first5_avg", "status", "error_message", "finding",
    "wandb_run_url", "hf_checkpoint_url",
]

# Loss arms we accept. Anything else is a hard error (no silent fallback).
VALID_LOSSES = {"importance_sampling", "grpo", "dapo", "gspo", "drgrpo"}
VALID_DIFFICULTY = {"easy", "medium", "hard"}


# ---------------------------------------------------------------------------
# Difficulty bucketing for GSM8K (reasoning-step terciles)
# ---------------------------------------------------------------------------
def _gsm8k_step_count(answer_rationale: str) -> int:
    """Cheap proxy for problem difficulty: number of reasoning steps in the
    GSM8K reference solution. GSM8K rationales put one step per line and embed
    calculator annotations like <<48/2=24>>. More steps ~= harder.
    """
    import re

    # Count calculator annotations; fall back to non-empty solution lines.
    calc = len(re.findall(r"<<[^>]*>>", answer_rationale))
    if calc:
        return calc
    lines = [ln for ln in answer_rationale.splitlines() if ln.strip()]
    # last line is the "#### <final>" answer marker; don't count it as a step
    return max(len(lines) - 1, 1)


def build_difficulty_loader(probe_module, difficulty: str):
    """Return a replacement for live_zvf_probe.load_gsm8k_examples that filters
    the GSM8K training pool to the requested difficulty tercile, then formats
    examples exactly as the original loader does (so downstream code is
    byte-for-byte compatible).
    """
    import random
    import re

    from datasets import load_dataset

    SYSTEM_PROMPT = probe_module.SYSTEM_PROMPT
    QUESTION_SUFFIX = probe_module.QUESTION_SUFFIX

    def loader(limit: int, seed: int):
        ds = load_dataset("openai/gsm8k", "main", split="train")
        scored = []  # (step_count, question, answer)
        for row in ds:
            match = re.search(r"####\s*([\-\d,\.]+)", row["answer"])
            if not match:
                continue
            answer = match.group(1).replace(",", "").strip()
            scored.append((_gsm8k_step_count(row["answer"]), row["question"], answer))

        if not scored:
            return []

        # Tercile split by step count -> easy / medium / hard.
        scored.sort(key=lambda x: x[0])
        n = len(scored)
        lo, hi = n // 3, (2 * n) // 3
        buckets = {
            "easy": scored[:lo],
            "medium": scored[lo:hi],
            "hard": scored[hi:],
        }
        chosen = buckets[difficulty] or scored  # never hand back an empty pool

        examples = []
        for _steps, question, answer in chosen:
            prompt = (
                f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                f"<|im_start|>user\n{question}{QUESTION_SUFFIX}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            examples.append((prompt, answer))
        rng = random.Random(seed)
        rng.shuffle(examples)
        return examples[:limit]

    return loader


# ---------------------------------------------------------------------------
# Loss-arm advantage surrogate (applied on top of the in-repo GRPO loop)
# ---------------------------------------------------------------------------
def install_loss_arm(probe_module, loss: str) -> None:
    """Adjust the existing runner's loss surrogate for the requested arm.

    The in-repo loop computes a per-group standardized advantage and a
    REINFORCE-style loss `-(adv * sum(logprob))`. That is the canonical GRPO /
    importance_sampling surrogate already used across the repo. For dapo / gspo
    we wrap probe_module.loss_fn with the arm-specific post-processing the loop
    supports. We do NOT fabricate metrics; we only change how the (real) loss is
    formed from the (real) advantages and (real) log-probs.
    """
    if loss in ("importance_sampling", "grpo"):
        return  # baseline surrogate == what the runner already does

    import torch

    advs = probe_module._ADVS  # the live advantage buffer the runner fills

    if loss == "dapo":
        # DAPO: asymmetric (clip-higher) advantage clipping, length-unbiased.
        clip_lo, clip_hi = -2.0, 3.0

        def loss_fn(_data, logprobs_list):
            losses = []
            for i in range(len(logprobs_list)):
                a = max(clip_lo, min(clip_hi, advs[i]))
                losses.append(-a * logprobs_list[i].sum())
            loss_t = losses[0] if len(losses) == 1 else torch.stack(losses).mean()
            return loss_t, {"loss": loss_t.item()}

    elif loss == "gspo":
        # GSPO: sequence-level objective -> aggregate log-probs by MEAN per
        # sequence (length-normalized) instead of SUM.
        def loss_fn(_data, logprobs_list):
            losses = []
            for i in range(len(logprobs_list)):
                lp = logprobs_list[i]
                seq = lp.mean() if lp.numel() > 0 else lp.sum()
                losses.append(-advs[i] * seq)
            loss_t = losses[0] if len(losses) == 1 else torch.stack(losses).mean()
            return loss_t, {"loss": loss_t.item()}
    elif loss == "drgrpo":
        # Dr.GRPO: remove the /std normalization of the per-group advantage.
        # The runner computes A = (r - mean) / std by default. Dr.GRPO uses
        # A = (r - mean) (no /std), which removes the bias toward shorter
        # responses. We achieve this by toggling the module-level flag
        # _ADV_NORMALIZE_STD in live_zvf_probe BEFORE the runner's next
        # optimizer step. We do NOT modify the loss_fn (the standard
        # REINFORCE/GRPO loss form is unchanged); we only change the
        # advantage value. loss_fn here is a no-op pass-through to keep
        # the install_loss_arm contract uniform.
        probe_module._ADV_NORMALIZE_STD = False

        def loss_fn(_data, logprobs_list):
            # REINFORCE-style: -adv * sum(logprobs) over the response span,
            # matching the in-repo loop's loss kernel.
            losses = []
            for i in range(len(logprobs_list)):
                losses.append(-advs[i] * logprobs_list[i].sum())
            loss_t = losses[0] if len(losses) == 1 else torch.stack(losses).mean()
            return loss_t, {"loss": loss_t.item()}
    else:  # pragma: no cover - guarded earlier
        return

    probe_module.loss_fn = loss_fn


# ---------------------------------------------------------------------------
# master_results.csv idempotent upsert
# ---------------------------------------------------------------------------
def upsert_master_row(result: dict, cell: dict) -> None:
    """Insert/replace the row for this cell in master_results.csv.

    Pulls every value from `result` (the real runner's output). The row's
    experiment_id is the cell tag, matching run_sweep.py's resume key.
    """
    row = {
        "group": "Tinker GRPO",
        "experiment_id": cell["tag"],
        "experiment_name": cell["tag"].replace("-", " ").replace("_", " "),
        "model": result.get("model", cell["model"]),
        "model_short": result.get("model_short", cell["model"].split("/")[-1].lower()),
        # method records the loss arm so the audit can stratify by it.
        "method": "GRPO" if cell["loss"] in ("grpo", "importance_sampling")
                  else cell["loss"].upper(),
        "task": result.get("task", cell["task"]),
        "platform": "tinker",
        "seed": result.get("seed", cell["seed"]),
        "steps_completed": len(result.get("step_log", []) or result.get("reward_trace", [])),
        "peak_reward": result.get("peak_reward"),
        "last10_avg": result.get("last10_avg"),
        "first5_avg": result.get("first5_avg"),
        "status": result.get("status", "failed"),
        "error_message": result.get("error"),
        "finding": None,
        "wandb_run_url": None,
        "hf_checkpoint_url": result.get("checkpoint"),
    }

    rows: list[dict] = []
    header_ok = False
    if MASTER_CSV.exists():
        try:
            with open(MASTER_CSV, newline="") as f:
                reader = csv.DictReader(f)
                header_ok = reader.fieldnames == MASTER_COLS
                if header_ok:
                    rows = [r for r in reader if r.get("experiment_id") != row["experiment_id"]]
        except Exception:
            header_ok = False

    if not header_ok:
        # Don't clobber an existing CSV with a different schema; just skip the
        # convenience upsert. The per-run JSON (the real artifact) is unaffected
        # and aggregate_results.py remains the canonical CSV builder.
        print(
            f"[cell_runner] WARN: master_results.csv schema mismatch; "
            f"skipping CSV upsert (JSON result still written). path={MASTER_CSV}",
            file=sys.stderr,
        )
        return

    rows.append(row)
    tmp = MASTER_CSV.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MASTER_COLS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in MASTER_COLS})
    os.replace(tmp, MASTER_CSV)


# ---------------------------------------------------------------------------
# Argument parsing — EXACTLY the flags run_sweep.py emits
# ---------------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Per-cell ZVF sweep shim -> live_zvf_probe.run (real Tinker GRPO).",
    )
    ap.add_argument("--tag", required=True, help="unique per-cell id (== experiment_id)")
    ap.add_argument("--model", required=True, help="HF model id, e.g. Qwen/Qwen3-8B")
    ap.add_argument("--task", required=True, help="training task (gsm8k)")
    ap.add_argument("--loss", required=True, choices=sorted(VALID_LOSSES),
                    help="loss/advantage surrogate arm")
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--group-size", dest="group_size", required=True, type=int,
                    help="GRPO group size G")
    ap.add_argument("--lr", required=True, type=float)
    ap.add_argument("--steps", required=True, type=int)
    ap.add_argument("--rank", required=True, type=int, help="LoRA rank")
    ap.add_argument("--difficulty", required=True, choices=sorted(VALID_DIFFICULTY))
    ap.add_argument("--out", required=True, help="path to write the result JSON")
    # Optional pass-throughs (sweep doesn't emit these today; sane defaults
    # mirror live_zvf_probe.parse_args()).
    ap.add_argument("--batch", type=int, default=2,
                    help="prompts per GRPO step (default 2, matches in-repo loops)")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", dest="top_p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--max-prompt-tokens", type=int, default=1024)
    ap.add_argument("--example-limit", type=int, default=256)
    ap.add_argument("--save-every", type=int, default=5)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.task != "gsm8k":
        # The in-repo Tinker runner only implements GSM8K today. Fail loudly
        # rather than silently train on the wrong data.
        print(
            f"[cell_runner] ERROR: task='{args.task}' is not supported by the "
            f"in-repo runner (only 'gsm8k'). Wire a new loader before using it.",
            file=sys.stderr,
        )
        return 2

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cell = {
        "tag": args.tag, "model": args.model, "task": args.task,
        "loss": args.loss, "seed": args.seed, "group_size": args.group_size,
        "lr": args.lr, "steps": args.steps, "rank": args.rank,
        "difficulty": args.difficulty,
    }

    # --- import the REAL runner (this is the only training code) -------------
    sys.path.insert(0, str(HERE))
    try:
        import live_zvf_probe as probe
    except Exception as exc:  # pragma: no cover - environment dependent
        # Import of the runner module itself failing (e.g. missing tinker SDK)
        # is a real, loud failure — write a failure stub, do NOT fake a result.
        stub = _failure_stub(cell, f"failed to import live_zvf_probe: {exc}")
        out_path.write_text(json.dumps(stub, indent=2) + "\n")
        print(f"[cell_runner] ERROR: {stub['error']}", file=sys.stderr)
        traceback.print_exc()
        return 1

    # --- wire mappings onto the real runner ----------------------------------
    probe.load_gsm8k_examples = build_difficulty_loader(probe, args.difficulty)
    install_loss_arm(probe, args.loss)

    # Build the Namespace the runner's run() expects. live_zvf_probe.run reads
    # exactly these attributes.
    run_args = argparse.Namespace(
        model=args.model,
        seed=args.seed,
        rank=args.rank,
        steps=args.steps,
        group=args.group_size,          # runner calls it `group`
        batch=args.batch,
        lr=args.lr,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        max_prompt_tokens=args.max_prompt_tokens,
        example_limit=args.example_limit,
        save_every=args.save_every,
        tag=args.tag,
    )

    print(
        f"[cell_runner] cell={args.tag} model={args.model} loss={args.loss} "
        f"G={args.group_size} difficulty={args.difficulty} steps={args.steps} "
        f"rank={args.rank} seed={args.seed} -> live_zvf_probe.run",
        flush=True,
    )

    # --- credential pre-check: fail loudly, never fake a completed result ----
    probe.load_env_file(probe.ROOT / ".env")
    if not os.environ.get("TINKER_API_KEY"):
        stub = _failure_stub(
            cell,
            "TINKER_API_KEY is not set and was not found in .env — refusing to "
            "fabricate a result. Set TINKER_API_KEY (and WANDB_API_KEY) and re-run.",
        )
        out_path.write_text(json.dumps(stub, indent=2) + "\n")
        try:
            upsert_master_row(stub, cell)
        except Exception:
            pass
        print(f"[cell_runner] ERROR: {stub['error']}", file=sys.stderr)
        return 1

    # --- run the REAL training cell ------------------------------------------
    try:
        result = probe.run(run_args)
    except Exception as exc:
        # The runner already wrote a failure JSON to its own results dir AND
        # re-raised. Mirror that failure to --out so the sweep ledger sees it;
        # do NOT invent a successful result.
        stub = _failure_stub(cell, f"runner raised: {exc}")
        stub["traceback"] = traceback.format_exc()
        out_path.write_text(json.dumps(stub, indent=2) + "\n")
        try:
            upsert_master_row(stub, cell)
        except Exception:
            pass
        print(f"[cell_runner] FAILED cell={args.tag}: {exc}", file=sys.stderr)
        return 1

    # --- success: stamp the cell's mapping metadata onto the real result -----
    # (We annotate provenance but never overwrite a measured metric.)
    result.setdefault("loss", args.loss)
    result.setdefault("difficulty", args.difficulty)
    result["sweep_cell_tag"] = args.tag
    result["loss_arm"] = args.loss
    result["difficulty_bucket"] = args.difficulty

    out_path.write_text(json.dumps(result, indent=2) + "\n")
    try:
        upsert_master_row(result, cell)
    except Exception as exc:
        print(f"[cell_runner] WARN: master CSV upsert failed: {exc}", file=sys.stderr)

    status = result.get("status", "unknown")
    print(f"[cell_runner] cell={args.tag} status={status} out={out_path}", flush=True)
    return 0 if status == "completed" else 1


def _failure_stub(cell: dict, error: str) -> dict:
    return {
        "tag": cell["tag"],
        "status": "failed",
        "model": cell["model"],
        "model_short": cell["model"].split("/")[-1].lower(),
        "task": cell["task"],
        "loss": cell["loss"],
        "loss_arm": cell["loss"],
        "difficulty": cell["difficulty"],
        "difficulty_bucket": cell["difficulty"],
        "seed": cell["seed"],
        "group_size": cell["group_size"],
        "lr": cell["lr"],
        "steps": cell["steps"],
        "rank": cell["rank"],
        "sweep_cell_tag": cell["tag"],
        "error": error,
        "failed_at": datetime.now(timezone.utc).isoformat(),
    }


if __name__ == "__main__":
    sys.exit(main())
