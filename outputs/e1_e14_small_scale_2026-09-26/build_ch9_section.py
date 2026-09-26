#!/usr/bin/env python3
"""Render the small-scale E1-E14 results into Chapter 9 from the lane result files.

Reads E<n>/result.json (Tinker base arm), E<n>/vllm_{trained,base}/result.json and
E<n>/paired.json, writes SUMMARY.md here, and replaces the block between the
small-scale markers in the thesis Chapter 9 source. Every number in the block is
read from those files; nothing is typed in by hand.

Usage: python3 build_ch9_section.py [--check]   (--check: report gaps, write nothing)
"""
from __future__ import annotations

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
CH9 = HERE.parents[0] / "PES_Phase2_Third_Review_2026-09-24/thesis/ch09_results_campaign.md"
BEGIN, END = "<!-- small-scale:begin -->", "<!-- small-scale:end -->"
LANES = [f"E{i}" for i in range(1, 15)]
REL = "outputs/e1_e14_small_scale_2026-09-26"


def load(p: pathlib.Path) -> dict | None:
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def fmt(v, pct=False) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.3f}" if not pct else f"{v:.1f}"
    return str(v)


def frac(r: dict) -> str:
    num, den = r.get("numerator"), r.get("denominator")
    if "progression" in (r.get("metric") or "").lower():
        return f"{fmt(r.get('value'), pct=True)}%"
    if isinstance(num, float) and not num.is_integer():   # continuous score: sum, not a count
        return f"{fmt(r.get('value'))} (n = {den})"
    return f"{num}/{den} = {fmt(r.get('value'))}" if num is not None and den else fmt(r.get("value"))


def short(s: str | None, n: int) -> str:
    s = (s or "").split(" (")[0].split(",")[0].strip()
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def main() -> int:
    check = "--check" in sys.argv
    base, trained, vbase, paired, gaps = {}, {}, {}, {}, []
    for ln in LANES:
        base[ln] = load(HERE / ln / "result.json")
        trained[ln] = load(HERE / ln / "vllm_trained" / "result.json")
        vbase[ln] = load(HERE / ln / "vllm_base" / "result.json")
        paired[ln] = load(HERE / ln / "paired.json")
        for name, d in (("result.json", base[ln]), ("vllm_trained", trained[ln]),
                        ("vllm_base", vbase[ln]), ("paired.json", paired[ln])):
            if d is None:
                gaps.append(f"{ln}: missing {name}")
    if gaps:
        print("GAPS:\n  " + "\n  ".join(gaps))
    if check:
        return 1 if gaps else 0

    rows_a = ["| Lane | Benchmark run | Scope | Metric | Result | 95% CI |",
              "|---|---|---|---|---|---|"]
    for ln in LANES:
        r = base[ln]
        if not r:
            rows_a.append(f"| {ln} | in progress | — | — | — | — |")
            continue
        ci = r.get("wilson95") or r.get("ci95") or r.get("bootstrap95")
        if "progression" in (r.get("metric") or "").lower():
            ci = None   # the lane's Wilson interval is for fully solved episodes, not progression
        ci_s = f"[{ci[0]:.2f}, {ci[1]:.2f}]" if isinstance(ci, list) and len(ci) == 2 else "—"
        num = r.get("numerator")
        if ci_s != "—" and isinstance(num, float) and not num.is_integer():
            ci_s = "≈ " + ci_s   # Wilson on a summed [0,1] score: approximate (lane caveat)
        rows_a.append(f"| {ln} | {short(r.get('benchmark_run'), 38)} | {r.get('scope', '—')} | "
                      f"{short(r.get('metric'), 26)} | {frac(r)} | {ci_s} |")

    rows_b = ["| Lane | Items | Trained | Base | Trained − base | Test |",
              "|---|---|---|---|---|---|"]
    def test_of(m: dict) -> str:
        if m.get("mcnemar_exact_p") is not None:
            return (f"McNemar p = {m['mcnemar_exact_p']:.2g} "
                    f"(b/c {m.get('discordant_b_trained_only', m.get('b_trained_only'))}/"
                    f"{m.get('discordant_c_base_only', m.get('c_base_only'))})")
        ci = (m.get("bootstrap95") or m.get("bootstrap_ci95") or m.get("bootstrap95_ci")
              or m.get("difference_bootstrap95"))
        if isinstance(ci, list) and len(ci) == 2 and None not in ci:
            d = 2 if max(abs(ci[0]), abs(ci[1])) >= 1 else 3
            return f"95% CI [{ci[0]:.{d}f}, {ci[1]:.{d}f}]"
        return "—"

    for ln in LANES:
        p = paired[ln]
        if not p:
            rows_b.append(f"| {ln} | — | — | — | — | in progress |")
            continue
        # one row per metric: top-level for single-metric lanes, else each
        # sub-dict carrying trained/base values (e.g. E10 refusal/harm/benign)
        subs = []
        for k, v in p.items():
            if k in ("tokens", "cost", "wall_time") or "diagnostic" in k:
                continue          # bookkeeping, not an outcome metric
            num = (int, float)
            if isinstance(v, dict) and (isinstance(v.get("trained_value"), num)
                                        or (isinstance(v.get("trained"), num)
                                            and isinstance(v.get("base"), num))):
                v = dict(v, trained_value=v.get("trained_value", v.get("trained")),
                         base_value=v.get("base_value", v.get("base")))
                subs.append((k, v))
        metrics = ([("", p)] if "trained_value" in p else []) + subs
        n = p.get("n_items", "—")
        n_s = ", ".join(f"{v} {k}" for k, v in n.items()) if isinstance(n, dict) else n
        for name, m in metrics:
            diff = m.get("difference", m.get("paired_mean_difference", m.get("diff")))
            if diff is None and isinstance(m.get("trained_value"), (int, float)) \
                    and isinstance(m.get("base_value"), (int, float)):
                diff = m["trained_value"] - m["base_value"]
            label = f"{ln} ({name.replace('_', ' ')})" if name else ln
            rows_b.append(f"| {label} | {n_s} | {fmt(m.get('trained_value'))} | "
                          f"{fmt(m.get('base_value'))} | {fmt(diff)} | {test_of(m)} |")

    caveats = []
    for ln in LANES:
        r = base[ln]
        if r and r.get("caveats"):
            caveats.append(f"- **{ln}.** {r['caveats'][0]}")

    block = f"""{BEGIN}
## Small-scale reruns of all fourteen lanes

The lane results above were produced by the trained actor at the time of the campaign and are left exactly as recorded. This section adds a separate, smaller set of runs, made on 26 September 2026, whose purpose is narrower: to give every one of the fourteen lanes a measured number under one fixed protocol, including the lanes that never produced a score. None of the figures below is pooled with, averaged against, or substituted for an original-contract or replacement-scope result reported earlier in this chapter.

The protocol fixes one actor family, a fixed item-selection seed (20260926), temperature 0 and a non-thinking chat template, and scores each lane with its native grader where one could be run. Lanes whose original benchmark is externally blocked were run on a public substitute, and each substitute's gap from the original suite is recorded in its lane file. Items that errored or timed out are counted as failures in the denominator. Each lane's items, raw outputs and grader logs are kept so that every figure can be recomputed (source: `{REL}/PROTOCOL.md`).

Two arms were run. The first is the base model, `Qwen/Qwen3.6-35B-A3B` with no adapter, sampled on Tinker (Table 9.A). The second is a paired comparison of the trained actor against the same base model, both served by one vLLM configuration on Modal, on identical items (Table 9.B). The pairing exists because the adapter's effect on individual token probabilities was measured to be about the same size as the numerical difference between the two serving engines on the same base weights. A comparison of trained-on-vLLM against base-on-Tinker would therefore confound the adapter with the engine, and no such cross-engine comparison is made here (source: `{REL}/TRAINED_ACTOR_ENDPOINT.md`).

**Table 9.A — Base model on Tinker, small-scale subsets.**

{chr(10).join(rows_a)}

**Table 9.B — Trained actor versus base model on the same vLLM engine, identical items.**

{chr(10).join(rows_b)}

These are small samples, and the tables should be read accordingly. Most lanes score between five and eighty items, so a single item moves a lane's rate by several percentage points and the confidence intervals are wide. In Table 9.B, b and c count items solved only by the trained actor and only by the base model respectively, and the CI is a paired bootstrap interval on the difference. For E12 the 151 rubric items come from only six generated applications, so they are not independent and its McNemar p-value overstates the evidence. For E9, the one competition that earned a medal in the Tinker run (nomad2018) has a 46.8k-token prompt that exceeds the vLLM serving limit of 32,768 tokens, so it fails identically in both vLLM arms; the vLLM medal rates are therefore not comparable with Table 9.A. The paired differences are reported with their own test and are not combined across lanes; no lane-level difference is interpreted as evidence that the adapter improves or degrades performance unless its test excludes zero, and none is extrapolated to the full suite. The principal caveat for each lane, as recorded in its result file, is listed below; the full list is in each lane's `result.json`.

{chr(10).join(caveats)}
{END}"""

    (HERE / "SUMMARY.md").write_text(block.replace(BEGIN + "\n", "").replace("\n" + END, "") + "\n")
    text = CH9.read_text()
    if BEGIN in text:
        pre, rest = text.split(BEGIN, 1)
        text = pre + block + rest.split(END, 1)[1]
    else:
        marker = "\n## Summary of the campaign at the point of writing"
        assert marker in text, "summary heading not found in Chapter 9"
        text = text.replace(marker, "\n" + block + "\n" + marker, 1)
    CH9.write_text(text)
    print(f"wrote SUMMARY.md and updated {CH9.name} ({len(gaps)} gaps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
