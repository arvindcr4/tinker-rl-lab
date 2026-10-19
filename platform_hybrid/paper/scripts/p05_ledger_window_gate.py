#!/usr/bin/env python3
"""Window/label gate for paper 05's training-reward ledger (tab:main_results).

Each retained numeric ledger row in platform_hybrid/paper/paper.tex carries a
trailing ``% gate:<id>`` marker. For every row spec below this script
(1) loads the named artifact record, (2) recomputes planned/completed/logged
step counts, peak, and the tail mean over the last min(10, logged) points,
(3) checks them against the declared values, and (4) checks that the marked
LaTeX row prints exactly the declared strings, including the window label.
A row whose logged trace has fewer than 10 points must not be labelled
"last-10". Exit status 1 on any mismatch.

Usage: python3 platform_hybrid/paper/scripts/p05_ledger_window_gate.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
TEX = REPO / "platform_hybrid/paper/paper.tex"
RES = "platform_hybrid/experiments/tinker-runs/results"
MASTER = "platform_hybrid/experiments/master_results.json"
CV2_A = f"{RES}/campaign_v2_20260419_091120.json"
CV2_B = f"{RES}/campaign_v2_20260419_111623.json"

# id: (artifact, selector, planned, completed, logged, peak_str, tail_str)
# selector: None = whole file is the record; ("tag", t) = results[] entry with tag t;
# ("experiment_id", e) = master experiments[] entry.
ROWS = {
    "scale_qwen3-8b": (f"{RES}/scale_gsm8k_qwen3-8b.json", None, 30, 30, 30, "62.5", "34.4"),
    "scale_qwen3.5-4b": (f"{RES}/scale_gsm8k_qwen3.5-4b.json", None, 30, 30, 30, "100", "85.0"),
    "scale_qwen3.5-27b": (f"{RES}/scale_gsm8k_qwen3.5-27b.json", None, 30, 10, 3, "75.0", "43.7"),
    "scale_qwen3-32b": (f"{RES}/scale_gsm8k_qwen3-32b.json", None, 30, 10, 3, "31.2", "25.0"),
    "scale_llama-8b-inst": (f"{RES}/scale_gsm8k_llama-8b-inst.json", None, 30, 30, 30, "100", "84.4"),
    "frontier_deepseek-v3.1": (f"{RES}/frontier_gsm8k_deepseek-v3.1.json", None, 20, 20, 20, "100", "85.0"),
    "frontier_nemotron-120b": (f"{RES}/frontier_gsm8k_nemotron-120b.json", None, 20, 20, 20, "87.5", "16.2"),
    "frontier_qwen3-235b": (f"{RES}/frontier_gsm8k_qwen3-235b.json", None, 20, 15, 4, "100", "100"),
    "moe_qwen3-30b-moe": (f"{RES}/moe_gsm8k_qwen3-30b-moe.json", None, 30, 20, 5, "50.0", "32.5"),
    "moe_qwen3-30b-inst": (f"{RES}/moe_gsm8k_qwen3-30b-inst.json", None, 30, 10, 3, "100", "100"),
    "cv2a_qwen3-8b-base": (CV2_A, ("tag", "w1_qwen3-8b-base"), 30, 30, 30, "100", "85.6"),
    "cv2b_qwen3-8b-base": (CV2_B, ("tag", "w1_qwen3-8b-base"), 30, 30, 30, "100", "98.8"),
    "cv2b_kimi-k2-thinking": (CV2_B, ("tag", "w1_kimi-k2-thinking"), 30, 30, 30, "100", "83.1"),
    "cv2b_kimi-k25": (CV2_B, ("tag", "w1_kimi-k25"), 30, 30, 30, "100", "76.2"),
    "cv2b_gpt-oss-120b": (CV2_B, ("tag", "w1_gpt-oss-120b"), 30, 30, 30, "100", "91.9"),
    "cv2b_qwen35-397b": (CV2_B, ("tag", "w1_qwen35-397b"), 30, 30, 30, "100", "91.9"),
    "cv2b_llama31-70b-base": (CV2_B, ("tag", "w1_llama31-70b-base"), 30, 30, 30, "75.0", "45.0"),
    "cv2b_deepseek-v31-base": (CV2_B, ("tag", "w1_deepseek-v31-base"), 30, 30, 30, "93.8", "65.6"),
    "cv2b_llama31-8b-base": (CV2_B, ("tag", "w1_llama31-8b-base"), 30, 30, 30, "31.2", "8.8"),
    "cv2b_G2": (CV2_B, ("tag", "w2_qwen3-8b_G2"), 30, 30, 30, "75.0", "35.0"),
    "cv2b_G4": (CV2_B, ("tag", "w2_qwen3-8b_G4"), 30, 30, 30, "87.5", "28.7"),
    "cv2b_G16": (CV2_B, ("tag", "w2_qwen3-8b_G16"), 30, 30, 30, "62.5", "32.5"),
    "cv2b_G32": (CV2_B, ("tag", "w2_qwen3-8b_G32"), 30, 30, 30, "67.2", "38.8"),
    "trl_qwen3-8b": (MASTER, ("experiment_id", "modal_trl_trl_qwen3_8b"), 30, 30, 30, "37.5", "5.0"),
    "trl_llama32-3b": (MASTER, ("experiment_id", "modal_trl_trl_llama32_3b"), 30, 30, 30, "100", "71.2"),
    "trl_llama32-1b": (MASTER, ("experiment_id", "modal_trl_trl_llama32_1b"), 30, 30, 30, "87.5", "36.2"),
    "tool_qwen3-32b": (f"{RES}/cross_tool_qwen3-32b.json", None, 30, 30, 30, "0", "0"),
    "tool_llama-8b-inst": (f"{RES}/cross_tool_llama-8b-inst.json", None, 30, 30, 30, "0", "0"),
}


def fmt(x: float) -> str:
    """Percent with one decimal (Python float formatting of the stored value); 100/0 bare."""
    v = 100.0 * x
    if abs(v - 100.0) < 1e-9:
        return "100"
    if abs(v) < 1e-9:
        return "0"
    return f"{v:.1f}"


def record(path: str, sel):
    data = json.loads((REPO / path).read_text())
    if sel is None:
        return data
    key, val = sel
    pool = data["results"] if key == "tag" else data["experiments"]
    matches = [r for r in pool if r.get(key) == val]
    if len(matches) != 1:
        raise ValueError(f"{path}: {len(matches)} records with {key}={val}")
    return matches[0]


def main() -> int:
    tex_lines = TEX.read_text().splitlines()
    errors: list[str] = []
    for rid, (path, sel, planned, completed, logged, peak_s, tail_s) in ROWS.items():
        rec = record(path, sel)
        trace = rec["reward_trace"]
        n = len(trace)
        got_planned = rec.get("steps", rec.get("num_steps"))
        got_completed = rec.get("steps_completed", got_planned)
        tail = trace[-min(10, n) :]
        checks = {
            "planned": (got_planned, planned),
            "completed": (got_completed, completed),
            "logged": (n, logged),
            "peak": (fmt(max(trace)), peak_s),
            "tail": (fmt(sum(tail) / len(tail)), tail_s),
        }
        for name, (got, want) in checks.items():
            if got != want:
                errors.append(f"{rid}: artifact {name}={got!r} but declared {want!r}")
        rows = [ln for ln in tex_lines if ln.rstrip().endswith(f"% gate:{rid}")]
        if len(rows) != 1:
            errors.append(f"{rid}: expected one tex row with marker, found {len(rows)}")
            continue
        row = rows[0]
        body = re.sub(r"\s*%\s*gate:\S+\s*$", "", row)
        cells = [c.strip() for c in body.split("&")]
        joined = " ".join(cells)
        steps_label = f"{planned}/{completed}/{logged}"
        for want in (steps_label, f"{peak_s}\\%", f"{tail_s}\\%"):
            if want not in joined:
                errors.append(f"{rid}: tex row lacks {want!r}: {row.strip()}")
        window = "last 10" if logged >= 10 else f"all {logged}"
        if window not in joined:
            errors.append(f"{rid}: tex row lacks window label {window!r}")
        if logged < 10 and re.search(r"last[- ]10", joined):
            errors.append(f"{rid}: {logged}-point trace labelled last-10")
    print(f"ROWS={len(ROWS)}")
    if errors:
        print("FAIL")
        print("\n".join(f"- {e}" for e in errors))
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
