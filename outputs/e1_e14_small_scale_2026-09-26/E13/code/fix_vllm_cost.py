#!/usr/bin/env python3
"""Post-process vllm_<arm>/result.json: cost -> vLLM tokens, active wall time (two segments: initial run until the
disk-full stop, and the resume), vLLM route caveats. Everything is recomputed from raw/shim_calls.jsonl."""
import json, sys
from datetime import datetime, timezone
from pathlib import Path

E = Path(__file__).resolve().parents[1]
ts = lambda p: datetime.strptime(p.read_text().strip(), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()
for arm in ("trained", "base"):
    A = E / f"vllm_{arm}"; R = A / "raw"
    res = json.loads((A / "result.json").read_text())
    calls = [json.loads(l) for l in open(R / "shim_calls.jsonl")]
    t0, t_res = ts(R / "started_utc.txt"), ts(R / "resumed_utc.txt")
    end = lambda cs: max(c["t"] + c.get("dt", 0) for c in cs)
    seg1 = end([c for c in calls if c["t"] < t_res]) - t0
    seg2 = end([c for c in calls if c["t"] >= t_res]) - t_res
    tok = sum(c["prompt_tokens"] + c["completion_tokens"] for c in calls)
    res["cost"] = {"tinker_tokens": 0, "vllm_tokens": tok, "vllm_prompt_tokens": sum(c["prompt_tokens"] for c in calls),
                   "vllm_completion_tokens": sum(c["completion_tokens"] for c in calls),
                   "colab_units": 0.0, "modal_usd": None,
                   "modal_note": "shared H200 endpoint; GPU bill reconciled by lead across lanes"}
    res["wall_time_s"] = round(seg1 + seg2)
    res["wall_time_segments_s"] = [round(seg1), round(seg2)]
    res["finished_utc"] = datetime.fromtimestamp(end(calls), timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    res["decoding"] = "raw /v1/completions with the identical chat-templated (enable_thinking=False) prompt token ids as the Tinker-base run"
    res["caveats"] = [c for c in res["caveats"] if "not comparable to or pooled" not in c] + [
        "Paired vLLM arm: compare only against the other vLLM arm (E13/paired.json), never against the Tinker-base arm.",
        "Run was interrupted by a host disk-full error; episodes without a completed native JSON were rerun from scratch "
        "with the same seed after restart. Tokens from the aborted partial episodes are included in cost.",
        "Arms ran concurrently; each local shim capped at 4 in-flight requests to its endpoint."]
    (A / "result.json").write_text(json.dumps(res, indent=1))
    print(arm, res["value"], res["wall_time_s"], tok)
