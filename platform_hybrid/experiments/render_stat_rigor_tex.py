"""Render only validated protocol-v2 independent-run results. Legacy statistical payloads are rejected; decisions use one authoritative correction family."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(
    0, str(next(p for p in _Path(__file__).resolve().parents if (p / "platform_hybrid").is_dir()))
)
from platform_hybrid.experiments.research_contracts import EvidenceUnavailable


import json
import math
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
PAYLOAD = ROOT / "experiments" / "stat_rigor_tables.json"
OUT_TEX = ROOT / "paper" / "sections" / "stat_rigor_updates.tex"


# Rows from paper Table 1 (tab:main_results) that have step-level traces.
TABLE1_KEYS = [
    ("scale_gsm8k_qwen3-8b", "Qwen3-8B", "GRPO (Tinker)"),
    ("scale_gsm8k_qwen3.5-4b", "Qwen3.5-4B", "GRPO (Tinker)"),
    ("scale_gsm8k_llama-8b-inst", "Llama-3.1-8B-Inst", "GRPO (Tinker)"),
    ("frontier_gsm8k_deepseek-v3.1", "DeepSeek-V3.1", "GRPO (Tinker)"),
    ("frontier_gsm8k_nemotron-120b", "Nemotron-120B", "GRPO (Tinker)"),
    ("campaign_v2_w1_qwen3-8b-base", "Qwen3-8B-Base", "GRPO (Tinker)"),
    ("campaign_v2_w1_gpt-oss-120b", "GPT-OSS-120B", "GRPO (Tinker)"),
    ("ppo_qwen3-8b", "Qwen3-8B", "PPO (Modal H100)"),
    ("ppo_llama-8b-inst", "Llama-3.1-8B-Inst", "PPO (Modal H100)"),
    ("cross_tool_llama-8b-inst", "Llama-3.1-8B-Inst", "GRPO (tool-use)"),
    ("cross_tool_qwen3-32b", "Qwen3-32B", "GRPO (tool-use)"),
]


def fmt_p(p: Optional[float]) -> str:
    if p is None or (isinstance(p, float) and not math.isfinite(p)):
        return "---"
    if p < 1e-3:
        return "$<$0.001"
    if p >= 0.9995:
        return "1.000"
    return f"{p:.3f}"


def fmt_num(x: Optional[float], decimals: int = 3, *, signed: bool = False) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "---"
    fmt = f"{{:+.{decimals}f}}" if signed else f"{{:.{decimals}f}}"
    return fmt.format(x)


def fmt_pct(x: Optional[float], decimals: int = 1, *, signed: bool = False) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "---"
    fmt = f"{{:+.{decimals}f}}" if signed else f"{{:.{decimals}f}}"
    return fmt.format(x) + "\\%"


def fmt_ci(ci: Optional[List[float]], decimals: int = 3, *, as_pct: bool = False) -> str:
    if not ci or any(x is None or not math.isfinite(x) for x in ci):
        return "---"
    if as_pct:
        return f"[{ci[0]:.{decimals}f}\\%, {ci[1]:.{decimals}f}\\%]"
    return f"[{ci[0]:.{decimals}f}, {ci[1]:.{decimals}f}]"


def sig_marker(p_bonf: Optional[float]) -> str:
    if p_bonf is None or not math.isfinite(p_bonf):
        return ""
    if p_bonf < 0.001:
        return "$^{***}$"
    if p_bonf < 0.01:
        return "$^{**}$"
    if p_bonf < 0.05:
        return "$^{*}$"
    return ""


# ──────────────────────────────────────────────────────────────────────────────
def render(payload: Dict) -> str:
    """Render measured results with one authoritative corrected decision."""
    if (
        payload.get("protocol_version") != "hybrid-research-v2"
        or payload.get("analysis_unit") != "independent_run"
    ):
        raise EvidenceUnavailable(
            "Legacy statistical payload is not eligible for empirical rendering"
        )
    comparisons = payload.get("comparisons", [])
    if not comparisons or payload.get("family_size") != len(comparisons):
        raise EvidenceUnavailable("Complete validated comparison family required")

    def escape(value):
        return (
            str(value)
            .replace("_", r"\_")
            .replace("%", r"\%")
            .replace("&", r"\&")
            .replace("#", r"\#")
        )

    lines = [
        "% Correction scope: " + payload["correction_scope"],
        r"\begin{tabular}{lrrrrl}",
        r"Hypothesis & Runs (A/B) & Effect & 95\% CI & Corrected p & Decision \\",
    ]
    for c in comparisons:
        if not c.get("run_ids") or not c.get("receipt_ids") or not c.get("hypothesis_id"):
            raise EvidenceUnavailable("Missing hypothesis/run/receipt provenance")
        if c["n"] != [len(ids) for ids in c["run_ids"]]:
            raise EvidenceUnavailable("Displayed sample counts disagree with contributors")
        low, high = c["ci95"]
        decision = "reject" if c["reject"] else "not rejected"
        lines.append(
            f"{escape(c['hypothesis_id'])} & {c['n'][0]}/{c['n'][1]} & {c['effect']:+.3f} & [{low:+.3f}, {high:+.3f}] & {c['p_corrected']:.4g} & {decision} "
            + r"\\"
        )
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("payload", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    text = render(json.loads(args.payload.read_text()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(f"[render] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
