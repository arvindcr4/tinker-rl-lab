#!/usr/bin/env python3
"""P1-A: completion-only mask diagnostic for the Tinker GRPO runner.

Reviewer objection #3: the runner's loss_fn uses `logprobs_list[i].sum()` over
the full sequence (prompt+completion). Canonical GRPO masks prompt tokens out.
Tinker's forward_backward_custom only allows loss_fn_inputs keys
{'target_tokens', 'weights'} so we use 'weights' as the completion mask:
  weights = 0.0 on prompt positions, 1.0 on completion positions.

Three formulations on the same data:
  A: full_sequence_sum       -- the current runner (weights=1 everywhere)
  B: completion_only_sum     -- canonical (weights=mask)
  C: completion_only_mean    -- length-normalized variant

Outputs:
  experiments/results/p1a_mask_test.json
  experiments/p1a_mask_test.md

Requires TINKER_API_KEY env.
"""

from __future__ import annotations

import json
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "experiments" / "results"
OUT_JSON = OUT_DIR / "p1a_mask_test.json"
OUT_MD = ROOT / "experiments" / "p1a_mask_test.md"

MODEL = "Qwen/Qwen3.5-4B"
RANK = 4
SEED = 20260422

PROMPT_SHORT = "Solve: 7 + 5 = ?\n"
RESPONSE_SHORT = "7 + 5 = 12. \\boxed{12}"
PROMPT_LONG = (
    "You are a math assistant. Work through each step carefully. "
    "Read the problem twice before answering. Show your reasoning. "
    "Put the final numerical answer inside \\boxed{}.\n\n"
    "Problem: What is seven plus five?\n\n"
)
RESPONSE_LONG = "Seven plus five equals twelve. \\boxed{12}"


def require_env():
    if not os.environ.get("TINKER_API_KEY"):
        raise RuntimeError("TINKER_API_KEY not set")


def build_loss_fns():
    import torch

    def full_sum(data, logprobs_list):
        # full sequence sum; ignore weights
        losses = [-lp.sum() for lp in logprobs_list]
        stk = torch.stack(losses)
        return stk.mean(), {
            "mean_loss": stk.mean().item(),
            "per_seq_loss": [float(x.detach()) for x in losses],
        }

    def comp_only_sum(data, logprobs_list):
        losses = []
        for lp, d in zip(logprobs_list, data):
            w = d.loss_fn_inputs["weights"].to_torch().to(lp.device)
            losses.append(-(lp * w).sum())
        stk = torch.stack(losses)
        return stk.mean(), {
            "mean_loss": stk.mean().item(),
            "per_seq_loss": [float(x.detach()) for x in losses],
        }

    def comp_only_mean(data, logprobs_list):
        losses = []
        for lp, d in zip(logprobs_list, data):
            w = d.loss_fn_inputs["weights"].to_torch().to(lp.device)
            denom = w.sum().clamp(min=1.0)
            losses.append(-(lp * w).sum() / denom)
        stk = torch.stack(losses)
        return stk.mean(), {
            "mean_loss": stk.mean().item(),
            "per_seq_loss": [float(x.detach()) for x in losses],
        }

    return full_sum, comp_only_sum, comp_only_mean


def make_data(tok, prompt, response):
    import tinker.types as T
    prompt_ids = tok.encode(prompt, add_special_tokens=False)
    resp_ids = tok.encode(response, add_special_tokens=False)
    full_ids = prompt_ids + resp_ids
    target_ids = full_ids[1:] + [0]
    # completion mask aligned to prediction positions (shifted like target_ids).
    raw_mask = [0.0] * len(prompt_ids) + [1.0] * len(resp_ids)
    mask = raw_mask[1:] + [0.0]
    datum = T.Datum(
        model_input=T.ModelInput.from_ints(full_ids),
        loss_fn_inputs={
            "target_tokens": T.TensorData(
                data=target_ids, dtype="int64", shape=[len(target_ids)]
            ),
            "weights": T.TensorData(
                data=mask, dtype="float32", shape=[len(mask)]
            ),
        },
    )
    return {
        "prompt_len": len(prompt_ids),
        "resp_len": len(resp_ids),
        "total_len": len(full_ids),
        "datum": datum,
    }


def main():
    require_env()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "tag": "p1a_mask_test",
        "model": MODEL,
        "rank": RANK,
        "seed": SEED,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "started",
    }
    OUT_JSON.write_text(json.dumps(result, indent=2) + "\n")
    t0 = time.time()

    try:
        import tinker
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
        svc = tinker.ServiceClient(base_url=None)
        tc = svc.create_lora_training_client(base_model=MODEL, rank=RANK)
        result["run_id"] = tc.model_id

        short = make_data(tok, PROMPT_SHORT, RESPONSE_SHORT)
        long_ = make_data(tok, PROMPT_LONG, RESPONSE_LONG)

        full_sum, comp_only_sum, comp_only_mean = build_loss_fns()
        data = [short["datum"], long_["datum"]]

        out_full = tc.forward_backward_custom(data=data, loss_fn=full_sum).result()
        out_comp_sum = tc.forward_backward_custom(data=data, loss_fn=comp_only_sum).result()
        out_comp_mean = tc.forward_backward_custom(data=data, loss_fn=comp_only_mean).result()

        result["losses"] = {
            "full_sequence_sum": dict(out_full.metrics),
            "completion_only_sum": dict(out_comp_sum.metrics),
            "completion_only_mean": dict(out_comp_mean.metrics),
        }
        result["sequence_shapes"] = {
            "short": {k: short[k] for k in ("prompt_len", "resp_len", "total_len")},
            "long": {k: long_[k] for k in ("prompt_len", "resp_len", "total_len")},
        }

        full_losses = out_full.metrics.get("per_seq_loss", [])
        comp_losses = out_comp_sum.metrics.get("per_seq_loss", [])
        attribution = []
        for i, (f, c) in enumerate(zip(full_losses, comp_losses)):
            denom = f if abs(f) > 1e-9 else 1e-9
            attribution.append({
                "seq": i,
                "full_loss": f,
                "completion_loss": c,
                "prompt_contribution": f - c,
                "prompt_frac_of_full": (f - c) / denom,
            })
        result["prompt_attribution"] = attribution

        avg_prompt_frac = (
            sum(abs(a["prompt_frac_of_full"]) for a in attribution) / len(attribution)
            if attribution else 0.0
        )
        verdict = "LEAK" if avg_prompt_frac > 0.05 else "CLEAN"
        result["verdict"] = {
            "avg_abs_prompt_frac_of_full_loss": avg_prompt_frac,
            "threshold": 0.05,
            "verdict": verdict,
        }
        result["status"] = "completed"
        result["wall_clock_sec"] = time.time() - t0
        result["completed_at"] = datetime.now(timezone.utc).isoformat()
        OUT_JSON.write_text(json.dumps(result, indent=2) + "\n")

        lines = [
            "# P1-A: completion-only mask diagnostic",
            "",
            "Model: `" + MODEL + "`, rank " + str(RANK) + ", seed " + str(SEED) + ".",
            "Two sequences: short prompt + long prompt, same response `\\boxed{12}`.",
            "",
            "## Sequence shapes",
            "",
            "| seq | prompt_len | resp_len | total_len |",
            "|---|---:|---:|---:|",
            "| short | " + str(short['prompt_len']) + " | " + str(short['resp_len']) + " | " + str(short['total_len']) + " |",
            "| long  | " + str(long_['prompt_len']) + " | " + str(long_['resp_len']) + " | " + str(long_['total_len']) + " |",
            "",
            "## Loss comparison (same data, three mask formulations)",
            "",
            "| formulation | mean_loss |",
            "|---|---:|",
            "| full_sequence_sum (current runner) | " + str(out_full.metrics.get('mean_loss', 'n/a')) + " |",
            "| completion_only_sum (canonical) | " + str(out_comp_sum.metrics.get('mean_loss', 'n/a')) + " |",
            "| completion_only_mean (length-norm) | " + str(out_comp_mean.metrics.get('mean_loss', 'n/a')) + " |",
            "",
            "## Prompt-gradient attribution",
            "",
            "| seq | full loss | completion loss | prompt contribution | prompt/full |",
            "|---|---:|---:|---:|---:|",
        ]
        for a in attribution:
            lines.append(
                "| " + str(a['seq']) + " | "
                + ("%.4f" % a['full_loss']) + " | "
                + ("%.4f" % a['completion_loss']) + " | "
                + ("%+.4f" % a['prompt_contribution']) + " | "
                + ("%+.3f" % a['prompt_frac_of_full']) + " |"
            )
        lines += [
            "",
            "## Verdict: **" + verdict + "**",
            "",
            "Average |prompt fraction of full-sequence loss|: "
            + ("%.3f" % avg_prompt_frac) + ". Threshold for CLEAN: < 0.05.",
            "",
            "Current runner uses full_sequence_sum. If the prompt fraction is",
            "non-trivial (>5%), the recorded GRPO loss is contaminated by",
            "gradient on prompt tokens, which canonical GRPO does not do.",
            "",
        ]
        OUT_MD.write_text("\n".join(lines) + "\n")
        print("wrote " + str(OUT_JSON))
        print("wrote " + str(OUT_MD))
        print("verdict: " + verdict + " (avg abs prompt frac = " + ("%.3f" % avg_prompt_frac) + ")")

    except Exception as exc:
        result["status"] = "failed"
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc()
        result["wall_clock_sec"] = time.time() - t0
        OUT_JSON.write_text(json.dumps(result, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
