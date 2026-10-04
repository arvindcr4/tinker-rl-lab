#!/usr/bin/env python3
"""Recompute the public thesis's reported numbers from their artifacts.

Every claim names tokens printed in a chapter, the artifact(s) they come from
and how they are checked. A token must appear verbatim in its chapter, so the
registry cannot drift from the manuscript. Methods:

  R  recomputed from artifact rows (counts, means, Wilson, McNemar, sign-flip, bootstrap)
  S  matched to a stored summary field of the artifact
  T  transcription: the token appears verbatim in the cited source document
  A  arithmetic from the printed counts only; the artifact is not in this checkout
  W  not checkable: the source is withheld or absent (listed, never counted as passing)

Writes a JSON report (every token, every artifact SHA-256) and the one-page
number-to-artifact appendix. No model, scorer or provider is run.
"""

from __future__ import annotations

import argparse
import base64
import glob
import csv
import hashlib
import itertools
import json
import math
import random
import re
import statistics
import subprocess
import tempfile
from collections import Counter
from functools import cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
THESIS = "reports/public_revision_2026-10-04/thesis"
RES = "platform_hybrid/experiments/results"
SEC = "platform_hybrid/paper/sections"
SMALL = "outputs/e1_e14_small_scale_2026-09-26"
FINISH = "outputs/finish_pending_2026-09-27"
LABBENCH = "outputs/public_portfolio_2026-09-05/labbench_final_diagnostics.json"
AGENTDOJO = "outputs/public_portfolio_2026-09-05/agentdojo_native_receipt.json"
E11_RECEIPT = "outputs/modal_e1_e14/2026-08-16/e11_full_receipt.json"
E11_BOOT = "platform_hybrid/experiments/scripts/e11_design_cluster_bootstrap.json"
E14_DISP = (
    "outputs/public_portfolio_2026-09-05/native_finish_v5/e14_official_collector/"
    "official_result01/native_dispositions.jsonl"
)
E1_DIR = "outputs/modal_e1_e14/2026-08-16/e1_swe_bench_pro_full/seed1818"
STATS = "outputs/PES_Phase2_Third_Review_2026-09-24/thesis_review_2026-09-27/stats_followup"
CAMPAIGN_MD = "outputs/e1_e14_results_2026-09-05/E1_E14_Results.md"
FINAL_MD = "outputs/E1_E14_FINAL_RESULTS_2026-09-19.md"
# in thesis order: Chapters 1, 4, 6-9, then Appendices A and F
CHAPTERS = {
    "ch01": "ch01_introduction.md",
    "ch04": "ch04_methodology.md",
    "ch06": "ch06_results_core.md",
    "ch07": "ch07_results_infra.md",
    "ch09": "ch09_results_campaign.md",
    "ch10": "ch10_synthesis_conclusions.md",
    "appA": "ch_back_run_registry.md",
    "ch08": "ch08_results_fraud.md",
}
Z95 = 1.959963984540054


# ---------------------------------------------------------------- helpers
@cache
def text(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


@cache
def J(rel: str) -> Any:
    return json.loads(text(rel))


@cache
def jsonl(rel: str):
    return [json.loads(line) for line in text(rel).splitlines() if line.strip()]


def tsv(rel: str):
    lines = [x for x in text(rel).splitlines() if x and not x.startswith("#")]
    return list(csv.DictReader(lines, delimiter="\t"))


@cache
def expand(pattern: str) -> list[str]:
    return sorted(Path(p).relative_to(ROOT).as_posix() for p in glob.glob(str(ROOT / pattern)))


@cache
def sha256(rel: str) -> str:
    """File SHA-256; for a glob, SHA-256 of the sorted 'sha  path' lines of its matches."""
    if any(c in rel for c in "*?["):
        lines = "".join(f"{sha256(p)}  {p}\n" for p in expand(rel))
        return hashlib.sha256(lines.encode()).hexdigest()
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


def wilson(k, n, pct=False):
    p = k / n
    centre = (p + Z95**2 / (2 * n)) / (1 + Z95**2 / n)
    half = Z95 * math.sqrt(p * (1 - p) / n + Z95**2 / (4 * n * n)) / (1 + Z95**2 / n)
    lo, hi = max(0.0, centre - half), min(1.0, centre + half)
    return (100 * lo, 100 * hi) if pct else (lo, hi)


def mcnemar(pairs):
    b = sum(t and not a for t, a in pairs)
    c = sum(a and not t for t, a in pairs)
    n = b + c
    return b, c, min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n)


def signflip_p(diffs):
    """Exact two-sided sign-flip p for the mean of paired differences."""
    obs = abs(sum(diffs))
    hits = sum(
        abs(sum(s * d for s, d in zip(signs, diffs))) >= obs - 1e-12
        for signs in itertools.product((1, -1), repeat=len(diffs))
    )
    return hits / 2 ** len(diffs)


def holm(ps):
    order = sorted(range(len(ps)), key=ps.__getitem__)
    out, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        out[i] = running
    return out


def fisher_ci(r, n):
    z, se = math.atanh(r), 1 / math.sqrt(n - 3)
    return math.tanh(z - Z95 * se), math.tanh(z + Z95 * se)


def pct(k, n):
    return 100 * k / n


def frac(k, n):
    return f"{k}/{n}"


def comma_frac(k, n):
    return f"{k:,}/{n:,}"


# ---------------------------------------------------------------- compute blocks
def bernoulli_null():
    runs = J(f"{RES}/groupsize_zvf_sweep.json")["runs"]
    out = []
    for g in (2, 4, 8, 16):
        steps = [s for r in runs if r["group_size"] == g for s in r["step_log"]]
        measured = J(f"{RES}/groupsize_zvf_sweep.json")["summary"][str(g)]["mean_zvf"]
        null = statistics.mean(s["mean_reward"] ** g + (1 - s["mean_reward"]) ** g for s in steps)
        out += [measured, null]
    return out


def m16():
    d = J(f"{STATS}/m16_summary.json")
    step = d["primary_step_unit_last10"]
    arms = ("aero", "areal", "gift")
    ps = [signflip_p(step[a]["step_deltas"]) for a in arms]
    adj = dict(zip(arms, holm(ps)))
    means = {a: statistics.mean(step[a]["step_deltas"]) for a in arms}
    ci = {a: step[a]["ci95_orig_seed20260704"] for a in arms}
    return [
        means["aero"],
        *ci["aero"],
        means["areal"],
        *ci["areal"],
        means["gift"],
        *ci["gift"],
        adj["aero"],
        adj["areal"],
        adj["gift"],
        d["secondary_prompt_step_unit_last10"]["aero"]["holm_p_signflip_2sided"],
        d["secondary_prompt_step_unit_last10"]["aero"]["n_prompt_steps"],
    ]


def c1_ledger():
    rows = list(csv.DictReader(text(f"{THESIS}/c1_case_ledger.csv").splitlines()))
    n = len(rows)
    events = sum(r["frozen_primary_event"] == "True" for r in rows)
    corr = sum(r["conservative_corroborated_numeric_recovery"] == "True" for r in rows)
    cat = Counter(r["second_exclusive_category"] for r in rows)
    disagree = sum(r["first_second_category_disagreement"] == "True" for r in rows)
    return {
        "n": n,
        "events": events,
        "corr": corr,
        "cat": cat,
        "disagree": disagree,
        "wilson": wilson(events, n, pct=True),
    }


@cache
def e14():
    rows = jsonl(E14_DISP)
    accepted = [r for r in rows if r.get("native_accepted")]
    correct = sum(bool(r.get("correct")) and r.get("native_accepted") for r in rows)
    capped = [r for r in rows if r["actor_usage"]["completion_tokens"] >= 2048]
    uncapped = [r for r in rows if r["actor_usage"]["completion_tokens"] < 2048]

    def ok(rs):
        return sum(bool(r.get("correct")) and bool(r.get("native_accepted")) for r in rs)

    return {
        "n": len(rows),
        "accepted": len(accepted),
        "correct": correct,
        "capped": len(capped),
        "capped_ok": ok(capped),
        "uncapped": len(uncapped),
        "uncapped_ok": ok(uncapped),
    }


@cache
def e11_boot():
    d = J(E11_BOOT)
    pairs = list(d["paired_verdicts"].values())
    cc = sum(p[0] for p in pairs)
    spec = sum(p[1] for p in pairs)
    sums = [p[0] + p[1] for p in pairs]
    rng = random.Random(d["seed"])
    stats_ = sorted(
        sum(rng.choices(sums, k=len(sums))) / (2 * len(sums)) for _ in range(d["replicates"])
    )
    r = d["replicates"]
    lo, hi = stats_[int(0.025 * r)], stats_[int(0.975 * r) - 1]
    return {
        "cc": cc,
        "spec": spec,
        "n": len(pairs),
        "lo": lo,
        "hi": hi,
        "stored": d["ci95"],
        "source_sha_ok": d["source_sha256"] == sha256(E11_RECEIPT),
    }


def e1_corrupt_patches():
    """Count submitted patches that `git apply --stat` rejects (it parses, never applies)."""
    cands = J(f"{E1_DIR}/candidates.json")
    with tempfile.TemporaryDirectory() as scratch:
        corrupt = sum(
            subprocess.run(
                ["git", "apply", "--stat"],
                input=c["model_patch"].encode(),
                capture_output=True,
                cwd=scratch,
                check=False,
            ).returncode
            != 0
            for c in cands
        )
    return corrupt, len(cands)


ACTOR_RECEIPT = "checkpoints/grpo/pavlov_portfolio_api_swegym_qwen36_20260809_seed809.json"
ACTOR_WANDB = "wandb/run-20260809_140744-bsv8vx04/files/wandb-summary.json"
E10_DIR = "outputs/public_portfolio_2026-09-05"
E10_BODIES = [
    f"{E10_DIR}/agentdojo_fast02_journals/result/native_driver_responses.jsonl",
    f"{E10_DIR}/agentdojo_fast02_journals/dispatch_result.json",
    f"{E10_DIR}/agentdojo_continuation01_journals/result/native_driver_responses.jsonl",
    f"{E10_DIR}/agentdojo_continuation01_journals/result/responses.jsonl",
]
E13_RAW = (
    "outputs/PES_Phase2_Review_2026-09-12/finish/e13_control/E13-native-20260912-01/out/native/"
    "journal/episodes/*/turn-*/attempt-*/response.raw"
)
# Published report paths: identities the public derivative withholds are replaced, hashes kept.
REDACT = {r"wandb/run-[\w-]+/": "wandb/[withheld actor-training run]/"}


def actor_trace_rows():
    r = J(ACTOR_RECEIPT)["result"]
    trace = r["reward_trace"]
    started = re.search(r"run-(\d{4})(\d{2})(\d{2})_", ACTOR_WANDB)
    return [
        "mean {:.6f} over the first 5 steps and {:.6f} over the last 10; reward zero on {} of {} steps; "
        "loss zero on {} of {} steps".format(
            statistics.mean(trace[:5]),
            statistics.mean(trace[-10:]),
            sum(x == 0 for x in trace),
            len(trace),
            r["zero_loss_steps"],
            len(trace),
        ),
        "started {}-{}-{}; runtime {:,} s".format(
            *started.groups(), J(ACTOR_WANDB)["_wandb"]["runtime"]
        )
        if started
        else "no start date",
    ]


def finish_reasons(rel):
    """finish_reason of every response body in a journal (raw bodies are base64) or dispatch record."""
    out = []
    if rel.endswith("dispatch_result.json"):
        return re.findall(r'"finish_reason": ?"([a-z_]+)"', json.dumps(J(rel)["responses"]))
    for line in text(rel).splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        body = (
            json.loads(base64.b64decode(rec["raw_body_base64"]))
            if rec.get("raw_body_base64")
            else rec["response"]
        )
        out += [c.get("finish_reason") for c in body.get("choices", [])]
    return out


def serving_rows():
    gens = [json.loads((ROOT / p).read_text()) for p in expand(f"{E1_DIR}/tasks/*/generation.json")]
    tinker = [g for g in gens if g.get("generation_backend") is None]
    vllm = [g for g in gens if g.get("generation_backend") == "modal_gpu_vllm_merged_peft"]

    def capped(gs):
        return sum((g.get("response_tokens") or 0) >= 8192 for g in gs)

    main = finish_reasons(E10_BODIES[0]) + finish_reasons(E10_BODIES[1])
    cont = finish_reasons(E10_BODIES[2]) + finish_reasons(E10_BODIES[3])
    e13 = Counter(
        m for p in expand(E13_RAW) for m in re.findall(r'"finish_reason": ?"([a-z_]+)"', text(p))
    )
    return [
        f"{capped(tinker)}/{len(tinker)} Tinker and {capped(vllm)}/{len(vllm)} vLLM",
        f"{(main + cont).count('length')} length finish in {len(main) + len(cont)} recorded response bodies "
        f"({len(main)} main run, {len(cont)} continuation)",
        f"{e13['length']} length finishes against {e13['stop']} stop finishes in {len(expand(E13_RAW))} recorded responses",
    ]


def table_8a(lane):
    d = J(f"{SMALL}/{lane}/result.json")
    k, n = d["numerator"], d["denominator"]
    return [frac(int(k), n), k / n, *wilson(k, n)]


def table_8b(lane):
    d = J(f"{SMALL}/{lane}/paired.json")
    rows = d["items"] if lane in ("E1", "E2") else d["per_item"]
    rows = list(rows.values()) if isinstance(rows, dict) else rows
    key = {"E1": "{}_resolved", "E2": "{}_passed"}.get(lane, "{}")
    t: list[float]
    b: list[float]
    if lane == "E9":
        t = [int(r["trained"]["any_medal"]) for r in rows]
        b = [int(r["base"]["any_medal"]) for r in rows]
    else:
        cast = float if lane == "E4" else int
        t = [cast(r[key.format("trained")]) for r in rows]
        b = [cast(r[key.format("base")]) for r in rows]
    mt, mb = statistics.mean(t), statistics.mean(b)
    out: list[float | str] = [len(rows), mt, mb, round(round(mt, 4) - round(mb, 4), 4)]
    if lane != "E4":
        bb, cc, p = mcnemar(list(zip(t, b)))
        out += [p, f"{bb}/{cc}"]
    return out


def app_diffs(e12):
    """Per-application trained-minus-base rubric pass rates (items are '<app>.<n>')."""
    apps: dict[str, list] = {}
    for item, r in e12["per_item"].items():
        apps.setdefault(item.split(".")[0], []).append(r)
    return [statistics.mean(r["trained"] - r["base"] for r in rs) for rs in apps.values()]


def env_mean_diff(e13):
    """Equal-environment mean of per-item progression differences, in points."""
    envs: dict[str, list] = {}
    for r in e13["items"]:
        envs.setdefault(r["id"].split("/")[0], []).append(r["trained"] - r["base"])
    return 100 * statistics.mean(statistics.mean(v) for v in envs.values())


def table_8b_secondary():
    e10, e12, e13, e4 = (J(f"{SMALL}/{lane}/paired.json") for lane in ("E10", "E12", "E13", "E4"))
    e4_rows = e4["per_item"].values() if isinstance(e4["per_item"], dict) else e4["per_item"]
    return [
        "95% CI [{:.6f}, {:.6f}]".format(*e10["harm_score"]["bootstrap95"]),
        "95% CI [{:.6f}, {:.6f}]".format(*e10["benign_score"]["bootstrap95"]),
        "app-level sign-flip p = {:.6f} (n = {})".format(
            signflip_p(app_diffs(e12)), len(app_diffs(e12))
        ),
        "95% CI [{:.6f}, {:.6f}]".format(*e13["difference_bootstrap95"]),
        f"recomputed {env_mean_diff(e13):.6f}",
        "({:.6f})".format(min(float(r["trained"]) - float(r["base"]) for r in e4_rows)).replace(
            "-", "−"
        ),
    ]


def table_8c(lane):
    d = J(f"{FINISH}/{lane}/result.json")
    n, graded = d["n_attempted"], d["n_graded"]
    if lane == "E13":
        return [n, graded, d["score"], d["score"] - Z95 * d["se"], d["score"] + Z95 * d["se"]]
    k = d[{"E1": "n_resolved", "E2": "n_correct"}.get(lane, "n_success")]
    return [n, graded, frac(k, n), k / n, *wilson(k, n)]


# ---------------------------------------------------------------- claim registry
# (group, chapter, tokens, method, artifacts, compute) -- compute returns one
# value per token: a number (compared at the token's printed precision) or a
# string (compared exactly), or None to skip a token that is context only.
def claims():
    ss = f"{SMALL}/{{}}/result.json"
    fin = [f"{FINISH}/{lane}/result.json" for lane in ("E1", "E2", "E5", "E6", "E9", "E13")]
    c1 = c1_ledger
    yield (
        "§7.1 algorithm η² and CIs",
        "ch07",
        ["0.0454", "0.0094", "0.1476", "0.0075", "0.0019", "0.0790"],
        "S",
        [f"{RES}/p5p8/p5_headline_cis_full.tsv"],
        lambda: [
            float(tsv(f"{RES}/p5p8/p5_headline_cis_full.tsv")[i][k])
            for i in (0, 1)
            for k in ("point", "ci_lo", "ci_hi")
        ],
    )
    yield (
        "§7.1 managed head-to-head",
        "ch07",
        ["0.744", "0.742", "0.723", "0.710", "0.578", "0.567", "0.511"],
        "T",
        [f"{SEC}/p5_evidence.tex"],
        None,
    )
    yield (
        "§7.1 jitter vs ε-ZVF",
        "ch07",
        ["95/600", "0.1583", "0/600", "0.158", "0.000"],
        "S",
        [f"{RES}/pcd_vs_zvf_tolerance_analysis.json"],
        lambda: (
            lambda t: [
                frac(t["exact_zero_before_count"], t["n_groups"]),
                t["exact_zero_before_fraction"],
                frac(t["exact_zero_after_count"], t["n_groups"]),
                t["tolerance_after_fraction"],
                t["exact_zero_after_fraction"],
            ]
        )(J(f"{RES}/pcd_vs_zvf_tolerance_analysis.json")),
    )
    yield (
        "§7.1 manifest-outcome Mantel",
        "ch07",
        [
            "+0.44",
            "+0.39",
            "+0.28",
            "4,753",
            "9,999",
            "+0.47",
            "+0.43",
            "+0.32",
            "+0.407",
            "+0.336",
        ],
        "S",
        [f"{STATS}/m18_mantel_summary.json"],
        lambda: (
            lambda m: [
                m["mantel"]["zvf"]["rho_full_4753_pairs"],
                m["mantel"]["pcd"]["rho_full_4753_pairs"],
                m["mantel"]["mean_reward"]["rho_full_4753_pairs"],
                f"{m['n_pairs_full']:,}",
                f"{m['n_permutations']:,}",
                *(
                    m["sensitivity_seed_collapsed"]["mantel"][k]["rho"]
                    for k in ("zvf", "pcd", "mean_reward")
                ),
                m["original_script_rerun_2000_sampled_pairs"]["spearman_hamming_vs_d_zvf"],
                m["original_script_rerun_2000_sampled_pairs"]["spearman_hamming_vs_d_pcd"],
            ]
        )(J(f"{STATS}/m18_mantel_summary.json")),
    )
    yield (
        "§7.2 registry stress test",
        "ch07",
        ["4,030", "800", "3,230"],
        "R",
        [f"{RES}/p5p8/registry_stress_summary.json"],
        lambda: (
            lambda s: [
                f"{s['n_categories'] * s['n_entries'] * s['n_mutations_per_category']:,}",
                f"{s['n_categories'] * s['n_entries'] * s['n_mutations_per_category'] - sum(c['n'] for c in s['per_category'].values()):,}",
                f"{sum(c['n'] for c in s['per_category'].values()):,}",
            ]
        )(J(f"{RES}/p5p8/registry_stress_summary.json")),
    )
    yield (
        "§7.2 registry coverage",
        "ch07",
        [
            "40/40",
            "76/80",
            "54/60",
            "41/60",
            "29/60",
            "38/120",
            "8/40",
            "95%",
            "90%",
            "68%",
            "48%",
            "32%",
            "20%",
            "0.65",
            "1.00",
        ],
        "A",
        [],
        lambda: (
            [None] * 7
            + [
                round(pct(76, 80)),
                round(pct(54, 60)),
                round(pct(41, 60)),
                round(pct(29, 60)),
                round(pct(38, 120)),
                round(pct(8, 40)),
                *wilson(7, 7),
            ]
        ),
    )
    yield (
        "§7.2 same-stack method deltas",
        "ch07",
        [
            "−0.014",
            "−0.023",
            "−0.005",
            "−0.020",
            "−0.032",
            "−0.008",
            "+0.016",
            "−0.007",
            "+0.040",
            "p = 0.047",
            "0.047",
            "0.23",
            "p = 0.20",
            "160",
        ],
        "R",
        [f"{STATS}/m16_summary.json"],
        lambda: (lambda v: v[:9] + [f"p = {v[9]:.6f}", v[10], v[11], f"p = {v[12]:.6f}", v[13]])(
            m16()
        ),
    )
    yield (
        "§7.1 Bernoulli null vs ZVF",
        "ch07",
        ["0.838", "0.828", "0.764", "0.709", "0.691", "0.597", "0.631", "0.522"],
        "R",
        [f"{RES}/groupsize_zvf_sweep.json"],
        bernoulli_null,
    )
    yield (
        "§7.3 escalation, Bayes cost",
        "ch07",
        ["1,723", "1,867", "92.3%", "144", "8,854", "10,240", "14%"],
        "A",
        [],
        lambda: [
            None,
            None,
            round(pct(1723, 1867), 1),
            1867 - 1723,
            None,
            None,
            round(100 * (1 - 8854 / 10240)),
        ],
    )
    yield (
        "§7.3 controller frontier",
        "ch07",
        [
            "0.657",
            "0.599",
            "0.738",
            "90.10",
            "89.40",
            "90.77",
            "86.68",
            "84.00",
            "80.00",
            "64.76%",
            "13.69%",
            "21.56%",
            "6.376",
            "0.797",
            "0.776",
            "0.818",
            "466.75",
            "454.25",
            "485.00",
            "0.630",
            "+0.0494",
            "+0.0481",
            "+0.0506",
        ],
        "T",
        [f"{SEC}/p7_controller.tex"],
        None,
    )
    yield (
        "§7.3 hindsight/regime arithmetic",
        "ch07",
        ["35.24%", "72.9%", "27.1%", "0.797"],
        "A",
        [],
        lambda: [100 - 64.76, None, 100 - 72.9, 6.376 / 8],
    )
    yield (
        "§7.4 V2 mixedness table",
        "ch07",
        [
            "33 / 461",
            "7.16%",
            "5.14-9.88%",
            "2 / 9",
            "22.22%",
            "6.32-54.74%",
            "31 / 452",
            "6.86%",
            "4.87-9.57%",
            "13/461 (2.82%)",
            "290/16,384 parser failures (1.77%)",
            "18/16,384 cap hits (0.11%)",
            "12/439 (2.73%)",
            "16,384",
        ],
        "A",
        [],
        lambda: [
            None,
            round(pct(33, 461), 2),
            "{:.6f}-{:.6f}%".format(*wilson(33, 461, pct=True)),
            None,
            round(pct(2, 9), 2),
            "{:.6f}-{:.6f}%".format(*wilson(2, 9, pct=True)),
            None,
            round(pct(31, 452), 2),
            "{:.6f}-{:.6f}%".format(*wilson(31, 452, pct=True)),
            f"13/461 ({pct(13, 461):.6f}%)",
            f"290/16,384 parser failures ({pct(290, 16384):.6f}%)",
            f"18/16,384 cap hits ({pct(18, 16384):.6f}%)",
            f"12/439 ({pct(12, 439):.6f}%)",
            f"{512 * 32:,}",
        ],
    )
    yield (
        "§7.5 C1 results table",
        "ch07",
        ["11 / 64", "17.19%", "9.88-28.21%", "3 / 64", "4.69%", "87 / 4,096", "64 / 87"],
        "R",
        [f"{THESIS}/c1_case_ledger.csv"],
        lambda: (
            lambda c: [
                f"{c['events']} / {c['n']}",
                round(pct(c["events"], c["n"]), 2),
                "{:.6f}-{:.6f}%".format(*c["wilson"]),
                f"{c['corr']} / {c['n']}",
                round(pct(c["corr"], c["n"]), 2),
                None,
                f"{c['n']} / 87",
            ]
        )(c1()),
    )
    yield (
        "§7.5 C1 review categories",
        "ch07",
        [
            "| Clear wrong reference final value | 16 |",
            "| Materially ambiguous | 19 |",
            "| Inconsistent question | 2 |",
            "| Total | 64 |",
        ],
        "R",
        [f"{THESIS}/c1_case_ledger.csv"],
        lambda: (
            lambda c: [
                f"| Clear wrong reference final value | {c['cat']['clear_wrong_gold']} |",
                f"| Materially ambiguous | {c['cat']['materially_ambiguous']} |",
                f"| Inconsistent question | {c['cat']['inconsistent_question']} |",
                f"| Total | {c['n']} |",
            ]
        )(c1()),
    )
    # ---------------------------------------------------------- chapter 8 (ch09 file)
    yield (
        "Table 8.1 training record",
        "ch09",
        ["G = 4; 2 prompts per step; 40 steps", "80 prompt draws", "320 completions"],
        "A",
        [],
        lambda: [None, f"{2 * 40} prompt draws", f"{4 * 2 * 40} completions"],
    )
    yield (
        "Table 8.1 reward trace",
        "ch09",
        [
            "mean 0.025 over the first 5 steps and 0.2425 over the last 10; reward zero on 22 of 40 steps; "
            "loss zero on 35 of 40 steps",
            "started 2026-08-09; runtime 1,934 s",
        ],
        "R",
        [ACTOR_RECEIPT, ACTOR_WANDB],
        actor_trace_rows,
    )
    yield (
        "Table 8.2 E1/E8/E11/E14",
        "ch09",
        [
            "476 Tinker",
            "255 Modal",
            "14 generation failures",
            "4 artefacts lost",
            "1,259/1,967 unparsed (64.0%)",
            "1,086/1,967 length-truncated (55.2%)",
            "150/312 extraction failures (48.1%)",
            "1,002 of 1,024",
            "2,982/4,428 responses at the 2,048-token cap (67.3%)",
            "34.5%",
            "85.9%",
        ],
        "R",
        [f"{E1_DIR}/receipt.json", LABBENCH, E11_RECEIPT, E14_DISP],
        lambda: (
            lambda e1, lb, e11, e: [
                f"{e1['sampling']['backend_counts']['tinker_remote']} Tinker",
                f"{e1['sampling']['backend_counts']['modal_gpu_vllm_merged_peft']} Modal",
                f"{e1['coverage']['generation_failures']} generation failures",
                f"{e1['coverage']['generation_artifact_losses']} artefacts lost",
                f"{lb['native_answer_parse_none']:,}/{lb['evaluated']:,} unparsed "
                f"({pct(lb['native_answer_parse_none'], lb['evaluated']):.6f}%)",
                f"{lb['finish_reasons']['length']:,}/{lb['evaluated']:,} length-truncated "
                f"({pct(lb['finish_reasons']['length'], lb['evaluated']):.6f}%)",
                f"{e11['sampling']['extraction_failures']}/{e11['dataset']['problem_count']} extraction "
                f"failures ({pct(e11['sampling']['extraction_failures'], 312):.6f}%)",
                f"{round(e11['cost']['response_tokens'] / 312):,} of 1,024",
                f"{e['capped']:,}/{e['n']:,} responses at the 2,048-token cap ({pct(e['capped'], e['n']):.6f}%)",
                round(pct(e["capped_ok"], e["capped"]), 1),
                round(pct(e["uncapped_ok"], e["uncapped"]), 1),
            ]
        )(J(f"{E1_DIR}/receipt.json"), J(LABBENCH), J(E11_RECEIPT), e14()),
    )
    yield (
        "Table 8.2 finish and cap counts",
        "ch09",
        [
            "40/476 Tinker and 22/255 vLLM",
            "1 length finish in 418 recorded response bodies (407 main run, 11 continuation)",
            "73 length finishes against 189 stop finishes in 262 recorded responses",
        ],
        "R",
        [f"{E1_DIR}/tasks/*/generation.json", *E10_BODIES, E13_RAW],
        serving_rows,
    )
    yield (
        "§8.2 E8 LAB-Bench",
        "ch09",
        ["450/1967 = 0.2288", "[0.211, 0.248]", "450/708 = 0.636", "[0.600, 0.670]", "1,517"],
        "R",
        [LABBENCH],
        lambda: (
            lambda d: [
                f"{d['correct']}/{d['evaluated']} = {d['correct'] / d['evaluated']:.6f}",
                "[{:.6f}, {:.6f}]".format(*wilson(d["correct"], d["evaluated"])),
                f"{d['correct']}/{d['evaluated'] - d['native_answer_parse_none']} = "
                f"{d['correct'] / (d['evaluated'] - d['native_answer_parse_none']):.6f}",
                "[{:.6f}, {:.6f}]".format(
                    *wilson(d["correct"], d["evaluated"] - d["native_answer_parse_none"])
                ),
                f"{d['incorrect']:,}"
                if sum(c["correct"] for c in d["category_counts"].values()) == d["correct"]
                else "category mismatch",
            ]
        )(J(LABBENCH)),
    )
    yield (
        "§8.2 E10 AgentDojo",
        "ch09",
        ["88/97 = 0.9072", "[0.833, 0.950]"],
        "R",
        [AGENTDOJO],
        lambda: (
            lambda d: [
                f"{d['utility_passes']}/{d['score_denominator']} = {d['score']:.6f}",
                "[{:.6f}, {:.6f}]".format(*wilson(d["utility_passes"], d["completed_episodes"])),
            ]
        )(J(AGENTDOJO)),
    )
    yield (
        "§8.2 E11 cluster bootstrap",
        "ch09",
        [
            "129 passes of 312 = 0.4135",
            "67/156 = 0.4295",
            "62/156 = 0.3974",
            "[0.343, 0.487]",
            "200,000 replicates",
            "312,622",
            "129/162 = 0.796",
            "129/311 = 0.4148",
        ],
        "R",
        [E11_RECEIPT, E11_BOOT],
        lambda: (
            lambda b, r: [
                f"{r['pass_at_1']['raw']['passes']} passes of {r['pass_at_1']['raw']['denominator']} = {r['score']:.6f}"
                if b["source_sha_ok"] and b["cc"] + b["spec"] == r["pass_at_1"]["raw"]["passes"]
                else "mismatch",
                f"{b['cc']}/{b['n']} = {b['cc'] / b['n']:.6f}",
                f"{b['spec']}/{b['n']} = {b['spec'] / b['n']:.6f}",
                "[{:.6f}, {:.6f}]".format(b["lo"], b["hi"]),
                f"{J(E11_BOOT)['replicates']:,} replicates",
                f"{r['cost']['response_tokens']:,}",
                f"129/{312 - r['sampling']['extraction_failures']} = {129 / (312 - r['sampling']['extraction_failures']):.6f}",
                f"129/311 = {129 / 311:.6f}",
            ]
        )(e11_boot(), J(E11_RECEIPT)),
    )
    yield (
        "§8.2 E14 denominators",
        "ch09",
        [
            "4,426 accepted rows of 4,428",
            "0.5131043831902395",
            "2,271/4,428 = 51.29 per cent",
            "[0.498, 0.528]",
            "1,029 cases (34.5%)",
            "1,242 of the 1,446 uncapped responses (85.9%)",
        ],
        "R",
        [E14_DISP],
        lambda: (
            lambda e: [
                f"{e['accepted']:,} accepted rows of {e['n']:,}",
                repr(e["correct"] / e["accepted"]),
                f"{e['correct']:,}/{e['n']:,} = {pct(e['correct'], e['n']):.6f} per cent",
                "[{:.6f}, {:.6f}]".format(*wilson(e["correct"], e["n"])),
                f"{e['capped_ok']:,} cases ({pct(e['capped_ok'], e['capped']):.6f}%)",
                f"{e['uncapped_ok']:,} of the {e['uncapped']:,} uncapped responses "
                f"({pct(e['uncapped_ok'], e['uncapped']):.6f}%)",
            ]
        )(e14()),
    )
    yield (
        "§8.3 E1 SWE-bench Pro",
        "ch09",
        ["2/731 = 0.00274", "0.274 per cent", "[0.08%, 0.99%]", "593 of the 713"],
        "R",
        [f"{E1_DIR}/receipt.json", f"{E1_DIR}/candidates.json"],
        lambda: (
            lambda r, h: [
                f"{r['coverage']['resolved']}/{r['coverage']['expected_tasks']} = {r['score']:.5f}",
                f"{r['score_percent']:.6f} per cent",
                "[{:.6f}%, {:.6f}%]".format(*wilson(2, 731, pct=True)),
                f"{h[0]} of the {h[1]}",
            ]
        )(J(f"{E1_DIR}/receipt.json"), e1_corrupt_patches()),
    )
    yield (
        "§8.3 original-contract partials",
        "ch09",
        ["0.8628", "0.3115", "0.050505", "0.079365"],
        "T",
        [CAMPAIGN_MD],
        None,
    )
    yield (
        "§8.3 E9 coverage",
        "ch09",
        ["40 of 75", "53.33 per cent"],
        "A",
        [],
        lambda: [None, f"{pct(40, 75):.6f} per cent"],
    )
    t8c = {
        "E1": ["| 190 | 57 |", "1/190 = 0.005", "[0.001, 0.029]"],
        "E2": ["| 45 | 45 |", "27/45 = 0.600", "[0.455, 0.730]"],
        "E5": ["| 97 | 72 |", "8/97 = 0.082", "[0.042, 0.154]"],
        "E6": ["| 812 | 704 |", "90/812 = 0.111", "[0.091, 0.134]"],
        "E9": ["| 34 | 34 |", "10/34 = 0.294", "[0.168, 0.462]"],
        "E13": ["| 255 | 255 |", "26.1%", "[22.6, 29.6]"],
    }

    def row_8c(lane):
        v = table_8c(lane)
        if lane == "E13":
            return [f"| {v[0]} | {v[1]} |", f"{v[2]:.6f}%", f"[{v[3]:.6f}, {v[4]:.6f}]"]
        return [f"| {v[0]} | {v[1]} |", f"{v[2]} = {v[3]:.6f}", f"[{v[4]:.6f}, {v[5]:.6f}]"]

    yield (
        "Table 8.C replacement scopes",
        "ch09",
        [t for row in t8c.values() for t in row],
        "R",
        fin,
        lambda: [t for lane in t8c for t in row_8c(lane)],
    )
    yield (
        "§8.4 replacement sensitivities",
        "ch09",
        ["between 0.111 and 0.244", "4/110", "8/72 = 0.111"],
        "R",
        fin[:1] + fin[2:4],
        lambda: (
            lambda e1, e5, e6: [
                f"between {e6['n_success'] / e6['n_attempted']:.6f} and "
                f"{(e6['n_success'] + e6['n_ungraded_judge_unavailable']) / e6['n_attempted']:.6f}",
                f"{e1['prior_waves_separate_not_pooled']['resolved']}/{e1['prior_waves_separate_not_pooled']['attempted']}",
                f"{e5['n_success']}/{e5['n_graded']} = {e5['n_success'] / e5['n_graded']:.6f}",
            ]
        )(J(fin[0]), J(fin[2]), J(fin[3])),
    )
    yield (
        "Table 8.D deviations",
        "ch09",
        ["| E1 |", "| E2 |", "| E5 |", "| E6 |", "| E9 |", "| E13 |"],
        "S",
        fin,
        lambda: [
            f"| {p.split('/')[-2]} |" if J(p).get("deviations") else "no deviations field"
            for p in fin
        ],
    )
    yield (
        "§8.5 E4 prior base run",
        "ch09",
        ["100 trials, one error and a mean reward of 0.0", "0.4087"],
        "S",
        [f"{SMALL}/E4/result.json"],
        lambda: (
            lambda d: [
                f"{d['prior_base_run']['n_trials']} trials, one error and a mean reward of "
                f"{d['prior_base_run']['mean_reward']}"
                if d["prior_base_run"]["n_errors"] == 1
                else "x",
                d["per_item_reward"]["btb-19b3361c"],
            ]
        )(J(f"{SMALL}/E4/result.json")),
    )
    t8a = {
        "E1": "0/10 = 0.000 | [0.00, 0.28]",
        "E2": "23/30 = 0.767 | [0.59, 0.88]",
        "E3": "4/8 = 0.500 | [0.22, 0.78]",
        "E5": "3/10 = 0.300 | [0.11, 0.60]",
        "E6": "15/36 = 0.417 | [0.27, 0.58]",
        "E7": "0/6 = 0.000 | [0.00, 0.39]",
        "E8": "26/80 = 0.325 | [0.23, 0.43]",
        "E9": "1/5 = 0.200 | [0.04, 0.62]",
        "E10": "20/30 = 0.667 | [0.49, 0.81]",
        "E11": "35/50 = 0.700 | [0.56, 0.81]",
        "E12": "122/151 = 0.808 | [0.74, 0.86]",
        "E14": "38/100 = 0.380 | [0.29, 0.48]",
    }
    yield (
        "Table 8.A base model on Tinker",
        "ch09",
        list(t8a.values()) + ["0.068 (n = 6; one nonzero task, 0.409)", "| 23.1% |"],
        "R",
        [ss.format(lane) for lane in (*t8a, "E4", "E13")],
        lambda: (
            ["{} = {:.6f} | [{:.6f}, {:.6f}]".format(*table_8a(lane)) for lane in t8a]
            + [
                (
                    lambda d: (
                        f"{d['value']:.6f} (n = {d['denominator']}; one nonzero task, "
                        f"{max(d['per_item_reward'].values()):.6f})"
                    )
                )(J(ss.format("E4"))),
                f"| {J(ss.format('E13'))['value']:.6f}% |",
            ]
        ),
    )
    t8b = {
        "E1": "| E1 | 10 | 0.000 | 0.000 | 0.000 | McNemar p = 1 (b/c 0/0) |",
        "E2": "| E2 | 30 | 0.700 | 0.800 | -0.100 | McNemar p = 0.25 (b/c 0/3) |",
        "E3": "| E3 | 8 | 0.625 | 0.375 | 0.250 | McNemar p = 0.5 (b/c 2/0) |",
        "E4": "| E4 | 6 | 0.066 | 0.093 | -0.027 |",
        "E5": "| E5 | 10 | 0.500 | 0.200 | 0.300 | McNemar p = 0.38 (b/c 4/1) |",
        "E6": "| E6 | 36 | 0.444 | 0.472 | -0.028 | McNemar p = 1 (b/c 1/2) |",
        "E7": "| E7 | 6 | 0.167 | 0.000 | 0.167 | McNemar p = 1 (b/c 1/0) |",
        "E8": "| E8 | 80 | 0.312 | 0.312 | 0.000 | McNemar p = 1 (b/c 6/6) |",
        "E9": "| E9 | 5 | 0.000 | 0.000 | 0.000 | McNemar p = 1 (b/c 0/0) |",
        "E11": "| E11 | 50 | 0.660 | 0.660 | 0.000 | McNemar p = 1 (b/c 2/2) |",
        "E12": "| E12 | 151 | 0.748 | 0.781 | -0.033 | McNemar p = 0.3 (b/c 5/10)",
        "E14": "| E14 | 100 | 0.430 | 0.440 | -0.010 | McNemar p = 1 (b/c 5/6) |",
    }

    def fmt_p(p):
        return f"{p:.2g}" if p < 1 else "1"

    def row_8b(lane):
        v = table_8b(lane)
        base = f"| {lane} | {v[0]} | {v[1]:.6f} | {v[2]:.6f} | {v[3]:.6f} |".replace(
            "| -0.000 |", "| 0.000 |"
        )
        if lane == "E4":
            return base
        tail = f" McNemar p = {fmt_p(v[4])} (b/c {v[5]})"
        return base + tail + ("" if lane == "E12" else " |")

    yield (
        "Table 8.B binary lanes",
        "ch09",
        list(t8b.values()),
        "R",
        [f"{SMALL}/{lane}/paired.json" for lane in t8b],
        lambda: [row_8b(lane) for lane in t8b],
    )
    yield (
        "Table 8.B secondary tests",
        "ch09",
        [
            "95% CI [-0.053, 0.078]",
            "95% CI [-0.050, 0.012]",
            "app-level sign-flip p = 0.25 (n = 6)",
            "95% CI [-2.73, 7.09]",
            "recomputed 1.8752",
            "(−0.160)",
        ],
        "R",
        [f"{SMALL}/{lane}/paired.json" for lane in ("E10", "E12", "E13", "E4")],
        table_8b_secondary,
    )
    # ---------------------------------------------------------- chapter 9 (ch10 file)
    yield (
        "§9.1 P2 ZVF and residual",
        "ch10",
        ["0.130", "0.190", "0.155", "0.1583", "r = +0.21", "[−0.31, +0.62], n = 17"],
        "R",
        [f"{RES}/tinker_gsm8k_zvf_s{s}.json" for s in (42, 123, 456)]
        + [f"{RES}/zvf_iter26_residual.tsv"],
        lambda: (
            lambda z, r: [
                *z,
                statistics.mean(z),
                f"r = +{float(r['residual_r']):.6f}",
                "[{:+.6f}, {:+.6f}], n = {}".format(
                    *fisher_ci(float(r["residual_r"]), int(r["n_pooled"])), r["n_pooled"]
                ).replace("-", "−"),
            ]
        )(
            [J(f"{RES}/tinker_gsm8k_zvf_s{s}.json")["overall_zvf"] for s in (42, 123, 456)],
            tsv(f"{RES}/zvf_iter26_residual.tsv")[0],
        ),
    )
    yield (
        "§9.1 iteration-136 contrast",
        "ch10",
        ["p = 0.031", "0.0625", "p to 0.25"],
        "A",
        [],
        lambda: [f"p = {1 / 32:.6f}", 2 / 32, f"p to {min(1, 8 / 32):.6f}"],
    )
    yield (
        "§9.1 campaign restatement",
        "ch10",
        [
            "88/97 = 0.907",
            "[0.833, 0.950]",
            "129/312 = 41.35%",
            "[34.3%, 48.7%]",
            "2,271/4,428 = 51.29%",
            "2/731 = 0.274%",
            "[0.08%, 0.99%]",
            "27/45 = 0.600",
            "8/97 = 0.082",
            "90/812 = 0.111",
            "10/34 = 0.294",
            "26.1%",
            "1/190",
            "4/110",
        ],
        "R",
        [AGENTDOJO, E11_BOOT, E14_DISP, f"{E1_DIR}/receipt.json", *fin],
        lambda: (
            lambda a, b, e, r: [
                f"{a['utility_passes']}/{a['score_denominator']} = {a['score']:.6f}",
                "[{:.6f}, {:.6f}]".format(*wilson(88, 97)),
                f"{b['cc'] + b['spec']}/312 = {pct(b['cc'] + b['spec'], 312):.6f}%",
                "[{:.6f}%, {:.6f}%]".format(100 * b["lo"], 100 * b["hi"]),
                f"{e['correct']:,}/{e['n']:,} = {pct(e['correct'], e['n']):.6f}%",
                f"{r['coverage']['resolved']}/{r['coverage']['expected_tasks']} = {r['score_percent']:.6f}%",
                "[{:.6f}%, {:.6f}%]".format(*wilson(2, 731, pct=True)),
                *(
                    f"{table_8c(lane)[2]} = {table_8c(lane)[3]:.6f}"
                    for lane in ("E2", "E5", "E6", "E9")
                ),
                f"{table_8c('E13')[2]:.6f}%",
                table_8c("E1")[2],
                (lambda p: f"{p['resolved']}/{p['attempted']}")(
                    J(fin[0])["prior_waves_separate_not_pooled"]
                ),
            ]
        )(J(AGENTDOJO), e11_boot(), e14(), J(f"{E1_DIR}/receipt.json")),
    )
    yield (
        "Table 9.1 denominators",
        "ch10",
        ["0.3115", "2/731, 90/812, 4,428"],
        "R",
        [CAMPAIGN_MD, f"{E1_DIR}/receipt.json", fin[3], E14_DISP],
        lambda: [
            "0.3115" if "0.3115" in text(CAMPAIGN_MD) else "absent",
            f"{J(f'{E1_DIR}/receipt.json')['coverage']['resolved']}/731, "
            f"{J(fin[3])['n_success']}/{J(fin[3])['n_attempted']}, {e14()['n']:,}",
        ],
    )
    yield (
        "Table 9.1 SDK versions",
        "ch10",
        ["0.16.1 to 0.30.0"],
        "C",
        [
            "platform_hybrid/paper/ethics_statement_anon.tex",
            "zvf-program/flagship/modal_tinker_openai_bridge.py",
        ],
        lambda: [[r"tinker==0\.16\.1", r"\"tinker==0\.30\.0\""]],
    )
    yield (
        "§9.2 limitations figures",
        "ch10",
        ["82.0%", "83.3%", "776 kWh", "296 kg", "99.9%", "73.4%"],
        "T",
        ["LIMITATIONS_AND_IMPACT.md"],
        None,
    )
    # ---------------------------------------------------------- earlier chapters
    yield (
        "Table 6.1 same-stack rerun",
        "ch06",
        [
            "0.200 → 0.245",
            "0.200 → 0.250",
            "0.200 → 0.195",
            "| 0.270 |",
            "| 0.275 |",
            "| 0.196 |",
            "+0.050 [+0.015, +0.085]",
            "p = 0.016",
            "−0.005 [−0.038, +0.028]",
            "(p = 0.69)",
            "TOST p = 0.13",
        ],
        "R",
        [f"{RES}/samestack_gsm8k_cot.json"],
        lambda: (
            lambda d: [
                *(
                    f"{d['summary'][a]['heldout_pre_mean']:.6f} → {d['summary'][a]['heldout_post_mean']:.6f}"
                    for a in ("grpo_g8", "grpo_g2", "ppo")
                ),
                *(
                    f"| {d['summary'][a]['last10_mean']:.6f} |"
                    for a in ("grpo_g8", "grpo_g2", "ppo")
                ),
                "{:+.6f} [{:+.6f}, {:+.6f}]".format(
                    d["contrasts"]["grpo_g8_minus_ppo"]["mean_diff"],
                    *d["contrasts"]["grpo_g8_minus_ppo"]["ci95"],
                ),
                f"p = {d['contrasts']['grpo_g8_minus_ppo']['p_two_sided']:.6f}",
                "{:+.6f} [{:+.6f}, {:+.6f}]".format(
                    d["contrasts"]["grpo_g8_minus_grpo_g2"]["mean_diff"],
                    *d["contrasts"]["grpo_g8_minus_grpo_g2"]["ci95"],
                ).replace("-", "−"),
                f"(p = {d['contrasts']['grpo_g8_minus_grpo_g2']['p_two_sided']:.6f})",
                f"TOST p = {d['contrasts']['grpo_g8_minus_grpo_g2']['tost_p']:.6f}",
            ]
        )(J(f"{RES}/samestack_gsm8k_cot.json")),
    )
    yield (
        "§6.4.1 group-size sweep",
        "ch06",
        ["0.838", "0.764", "0.691", "0.631"],
        "S",
        [f"{RES}/groupsize_zvf_sweep.json"],
        lambda: [
            J(f"{RES}/groupsize_zvf_sweep.json")["summary"][g]["mean_zvf"]
            for g in ("2", "4", "8", "16")
        ],
    )
    yield (
        "Table F.4 sensor ablation CIs",
        "ch08",
        [
            "−0.0245 [−0.0320, −0.0174]",
            "−0.0135 [−0.0160, −0.0109]",
            "+0.0109 [0.0099, 0.0120]",
            "+0.0002 [−0.0002, 0.0007]",
        ],
        "R",
        [f"{RES}/p5p8/p8_headline_cis.tsv"],
        lambda: (
            lambda t: [
                "{:+.6f} [{:+.6f}, {:+.6f}]".format(-t[1][0], -t[1][2], -t[1][1]).replace("-", "−"),
                "{:+.6f} [{:+.6f}, {:+.6f}]".format(-t[3][0], -t[3][2], -t[3][1]).replace("-", "−"),
                "{:+.6f} [{:.6f}, {:.6f}]".format(*t[4]),
                "{:+.6f} [{:+.6f}, {:.6f}]".format(*t[0]).replace("-", "−"),
            ]
        )(
            [
                [float(r[k]) for k in ("point_estimate", "ci_lo_025", "ci_hi_975")]
                for r in tsv(f"{RES}/p5p8/p8_headline_cis.tsv")
            ]
        ),
    )
    yield (
        "Table A.2 cross-framework runs",
        "appA",
        [
            "last10 = 0.85625 — **real run**",
            "last10 = 0.05, 735.7 s — **real run**",
            "last10 = 0.5528 — **seeded dryrun",
            "last10 = 0.4785 — **seeded dryrun",
        ],
        "S",
        [f"{RES}/framework_comparison.json"],
        lambda: [
            (
                f"last10 = {f['last10_avg']:.6f}"
                + (f", {f['duration_s']:.6f} s" if "duration_s" in f else "")
                + (" — **real run**" if f["mode"] == "real" else " — **seeded dryrun")
            )
            for f in J(f"{RES}/framework_comparison.json")["frameworks"]
        ],
    )
    yield (
        "Table F.2 head-to-head",
        "ch08",
        ["| 10,000 | n/a | 0.7955 |", "| 500 | 0.7920 | 0.4827 |"],
        "S",
        [f"{RES}/quick_20260704/qp8_fraud.tsv"],
        lambda: [
            f"| {int(r['n']):,} | {'n/a' if r['accuracy'] == 'NA' else r['accuracy']} | {r['auc']} |"
            for r in tsv(f"{RES}/quick_20260704/qp8_fraud.tsv")
        ],
    )


# ---------------------------------------------------------------- front matter, method and appendix tables
AUDIT = "zvf-program/audit/results/full"
M = "platform_hybrid/experiments/modal"
TR = "platform_hybrid/experiments/tinker-runs"
SEEDS11 = (11, 23, 37, 53, 71, 89, 107, 131)
MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
NOTES = {
    "Table A.5 evidence tiers": "claim_to_run_table.tsv records P1-C2 as X; the thesis rectifies it to C "
    "because the Nemotron identity was arbitrated against the HF adapter (Table A.5 P1-C2 row).",
}


def rx(pattern, string, group=0) -> str:
    m = re.search(pattern, string)
    if m is None:
        raise ValueError(f"pattern not found: {pattern}")
    return m[group]


def audit_arm(arm):
    return {s: J(f"{AUDIT}/{arm}-seed-{s}.json") for s in SEEDS11}


def paired_t(diffs):
    from scipy import stats

    n = len(diffs)
    m, sd = statistics.mean(diffs), statistics.stdev(diffs)
    half = stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n)
    return m, m - half, m + half


def p11():
    grpo, dapo = audit_arm("grpo"), audit_arm("dapo")
    m, lo, hi = paired_t([dapo[s]["heldout_score"] - grpo[s]["heldout_score"] for s in SEEDS11])

    def avg(arm, key):
        return statistics.mean(r[key] for r in arm.values())

    return {
        "d": m,
        "lo": lo,
        "hi": hi,
        "roll": (avg(dapo, "rollouts"), avg(grpo, "rollouts")),
        "wall": avg(dapo, "wall_clock_seconds") / avg(grpo, "wall_clock_seconds"),
        "zvf": (avg(dapo, "mean_zvf"), avg(grpo, "mean_zvf")),
    }


def claim_rows():
    return list(
        csv.DictReader(
            text(f"{RES}/claim_to_run/claim_to_run_table.tsv").splitlines(), delimiter="\t"
        )
    )


def mean_gain(rows, pred=lambda r: True):
    return statistics.mean(r["heldout_gain"] for r in rows if pred(r))


def git_tag(tag):
    def run(*args):
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, cwd=ROOT, check=True
        ).stdout.strip()

    h, d = (
        run("rev-parse", "--short=8", f"{tag}^{{commit}}"),
        run("log", "-1", "--format=%ad", "--date=short", tag),
    )
    return f"commit `{h}` dated {int(d[8:])} {MONTHS[int(d[5:7]) - 1]} {d[:4]}"


def code_rows():
    """Table 4.2: (row tail as printed, runner, regexes that must all match the runner)."""
    t42 = "| 32 | 1e-5 (actor); 3e-5 default | 4 (actor); 8 default | 40 (actor) |"
    trl = "| 32 in code, 16 in dump | 1e-5 | 8 | 30 |"
    dumps = "platform_hybrid/experiments/framework_config_dumps"
    return [
        (
            t42,
            "platform_tinker/tinkerrl/grpo.py",
            [
                r"lora_rank: int = 32",
                r"lr: float = 3e-5",
                r"group_size: int = 8",
                r"eps: float = 1e-8",
            ],
        ),
        (
            t42,
            "platform_tinker/tinkerrl/grpo_cli.py",
            [r'"lora_rank": 32', r'"steps": 40', r'"group_size": 4', r'"lr": 1e-5'],
        ),
        (
            "| 32 | 3e-5 | 8 | 20–30 logged |",
            f"{TR}/scripts/tinker_parallel_runner.py",
            [r"rank=32", r"lr=3e-5", r"group=8"],
        ),
        (
            "| n/a | n/a | 8, T = 1.0 | 0 |",
            "platform_hybrid/experiments/tinker_direct_eval.py",
            [r"group_size: int = 8", r"temperature: float = 1\.0"],
        ),
        (
            "| 32 | 1e-5 | 8 | 40 |",
            f"{TR}/scripts/n2_reward_tensor_20260704.py",
            [
                r'"--rank", type=int, default=32',
                r'"--steps", type=int, default=40',
                r'"--group", type=int, default=8',
                r'"--lr", type=float, default=1e-5',
            ],
        ),
        (
            "| 16 (q, v) | 1e-4 | 2, 4, 8, 16 | 40 |",
            f"{M}/modal_groupsize_zvf_sweep.py",
            [
                r"LoraConfig\(r=16",
                r"lr=1e-4",
                r"GROUP_SIZES = \[2, 4, 8, 16\]",
                r"N_STEPS = 40",
                r"EPS = 1e-6",
            ],
        ),
        (
            "| 4 | 3e-5 | 2 / 16 | 160 / 20 |",
            f"{TR}/live_zvf_probe.py",
            [r'"--rank", type=int, default=4', r'"--lr", type=float, default=3e-5'],
        ),
        (
            "| 8 | 3e-5 | 2–16 | 8 |",
            "platform_hybrid/experiments/openings/groupsize_zvf.py",
            [
                r'"--rank", type=int, default=8',
                r'"--lr", type=float, default=3e-5',
                r'"--steps", type=int, default=8',
                r'"--groups", default="2,4,8,16"',
            ],
        ),
        (
            "| 16 | 1e-4 | 8 | 40 |",
            f"{M}/modal_drgrpo_vs_grpo.py",
            [r"LoraConfig\(r=16", r"lr=1e-4", r"GROUP = 8", r"N_STEPS = 40", r"K_EPOCHS = 2"],
        ),
        (
            "| 16 | 1e-5 | 8 | 30 |",
            f"{M}/modal_drgrpo_gsm8k_cot.py",
            [r"LoraConfig\(r=16", r"lr=1e-5", r"GROUP = 8", r"N_STEPS = 30", r"K_EPOCHS = 2"],
        ),
        (
            "| 4 | 3e-5 | 8 | 30 |",
            f"{TR}/live_zvf_probe.py",
            [
                r'"--rank", type=int, default=4',
                r'"--lr", type=float, default=3e-5',
                r'choices=\["grpo", "drgrpo"\]',
            ],
        ),
        (
            "| 16 (q, v) | 1e-4 | GRPO 16 prompts × 8; PPO 128 prompts × 1 | 40 |",
            f"{M}/modal_samestack_ppo_grpo.py",
            [
                r"LoraConfig\(r=16",
                r'target_modules=\["q_proj", "v_proj"\]',
                r"lr=1e-4",
                r"N_GEN = 128",
                r"GROUP = 8",
                r"N_STEPS = 40",
            ],
        ),
        (
            "| full fine-tune | 2e-6 | 4, adaptive up to 10 | 10 |",
            "zvf-program/colab-experiments/e3_open_audit.py",
            [
                r'"--learning-rate", type=float, default=2e-6',
                r'"--gmax", type=int, default=10',
                r'"--steps", type=int, default=10',
                r"asymmetric clip \[0\.2, 0\.28\]",
            ],
        ),
        (
            "| 32 | 1e-5 | 4 → 6 → 8 | 16 |",
            f"{TR}/scripts/qp7_adaptive_g_20260704.py",
            [
                r'"--rank", type=int, default=32',
                r'"--lr", type=float, default=1e-5',
                r'"--steps", type=int, default=16',
                r"G_LADDER = \[4, 6, 8\]",
            ],
        ),
        (
            "| 16 (dump) | 1e-5 | 8 | 30 |",
            f"{dumps}/tinker_qwen3_8b_gsm8k.yaml",
            [r"rank: 16", r"lr: 1\.0e-5", r"group_size: 8", r"beta: null"],
        ),
        (
            trl,
            f"{M}/modal_grpo_trl.py",
            [r"r=32", r"learning_rate=1e-5", r"num_generations=8", r"max_steps=30"],
        ),
        (trl, f"{dumps}/trl_qwen3_8b_gsm8k.yaml", [r"rank: 16", r"beta: 0\.04"]),
        (
            "| 32 | 1e-5 | n/a | not recorded |",
            f"{M}/modal_ppo_campaign.py",
            [r"r=32", r"learning_rate=1e-5", r"target_kl="],
        ),
    ]


def ch01_rows():
    sweep = J(f"{RES}/groupsize_zvf_sweep.json")["summary"]
    gs = ("2", "4", "8", "16")
    zvf = [J(f"{RES}/tinker_gsm8k_zvf_s{s}.json")["overall_zvf"] for s in (42, 123, 456)]
    pp = J(f"{RES}/samestack_ppo_grpo.json")
    cot = J(f"{RES}/samestack_gsm8k_cot.json")
    c, g = cot["contrasts"]["grpo_g8_minus_ppo"], cot["contrasts"]["grpo_g8_minus_grpo_g2"]
    dv, dc = J(f"{RES}/drgrpo_vs_grpo.json"), J(f"{RES}/drgrpo_gsm8k_cot.json")
    fw = {x["framework"]: x for x in J(f"{RES}/framework_comparison.json")["frameworks"]}
    co = J(f"{RES}/curriculum_opening/results.json")
    p = p11()
    arm = {a: [r["heldout_acc"] for r in pp["runs"] if r["algo"] == a] for a in ("grpo", "ppo")}
    return [
        "{:.6f} / {:.6f} / {:.6f}; pooled {:.6f}".format(*zvf, statistics.mean(zvf)),
        "{:.6f} / {:.6f} / {:.6f} / {:.6f} at G = 2 / 4 / 8 / 16 (fall of {:.6f})".format(
            *(sweep[x]["mean_zvf"] for x in gs), sweep["2"]["mean_zvf"] - sweep["16"]["mean_zvf"]
        ),
        "{:.6f} / {:.6f} / {:.6f} / {:.6f} at G = 2 / 4 / 8 / 16".format(
            *(sweep[x]["heldout_acc_mean"] for x in gs)
        ),
        "GRPO {:.6f}, PPO {:.6f}; paired Δ = {:.6f}, p = {:.6f}".format(
            statistics.mean(arm["grpo"]),
            statistics.mean(arm["ppo"]),
            statistics.mean(arm["grpo"]) - statistics.mean(arm["ppo"]),
            pp["paired_grpo_vs_ppo"]["p_two_sided"],
        ),
        "GRPO (G = 8) {:.6f}, GRPO (G = 2) {:.6f}, PPO {:.6f}; GRPO G8 − PPO {:+.6f} [{:+.6f}, {:+.6f}], "
        "paired-t p = {:.6f}, exact p = {:.6f}; G8 − G2 {:+.6f} [{:+.6f}, {:+.6f}]".format(
            *(cot["summary"][a]["heldout_post_mean"] for a in ("grpo_g8", "grpo_g2", "ppo")),
            c["mean_diff"],
            *c["ci95"],
            c["p_two_sided"],
            signflip_p(c["diffs"]),
            g["mean_diff"],
            *g["ci95"],
        ),
        "{:.6f} vs {:.6f}, p = {:.6f}; completions {:.6f} tokens".format(
            dv["summary"]["grpo"]["heldout_mean"],
            dv["summary"]["dr_grpo"]["heldout_mean"],
            next(v for k, v in dv["paired_drgrpo_vs_grpo"].items() if k.startswith("p_two")),
            dv["summary"]["grpo"]["mean_comp_len"],
        ),
        "{:.6f} → {:.6f} / {:.6f} → {:.6f}".format(
            *(
                dc["summary"][a][k]
                for a in ("grpo", "dr_grpo")
                for k in ("heldout_pre_mean", "heldout_post_mean")
            )
        ),
        "Δ = {:+.6f}, paired-t 95% CI [{:+.6f}, {:+.6f}]".format(p["d"], p["lo"], p["hi"]),
        "{:.6f} (Tinker, Qwen3-8B-Base) vs {:.6f} (TRL".format(
            fw["Tinker"]["last10_avg"], fw["TRL"]["last10_avg"]
        ),
        "{:+.6f} in both arms".format(co["baseline"]["heldout_gain"])
        if co["baseline"]["heldout_gain"] == co["curriculum"]["heldout_gain"]
        else "arms differ",
    ]


def a6_rows():
    camp, tb, hc, p4 = (
        J(f"{RES}/{n}/results.json")
        for n in ("campaign", "token_budget", "hard_curriculum", "p4_surprise")
    )
    co = J(f"{RES}/curriculum_opening/results.json")

    def pick(rows, prefix):
        return [r for r in rows if r.get("name", "").startswith(prefix)]

    def span(rows, key):
        return min(r[key] for r in rows), max(r[key] for r in rows)

    b4, c4 = pick(camp, "baseline-G4"), pick(camp, "curriculum-G4")
    tbb, tbc, hcb, hcc = (
        pick(tb, "baseline"),
        pick(tb, "curriculum"),
        pick(hc, "baseline"),
        pick(hc, "curriculum"),
    )
    single = {r["G"]: r for r in camp if not r["name"].startswith(("baseline-G4", "curriculum-G4"))}
    n_runs = len(camp) + 2 + len(tb) + len(hc) + len(p4) + 1 + 4 + 4 + 2
    return [
        "{} runs; mean Δ {:+.6f} ({:.6f}–{:.6f})".format(
            len(b4), mean_gain(b4), *span(b4, "heldout_gain")
        ),
        "{} runs; mean Δ {:+.6f} ({:+.6f} to {:+.6f}); oversample {:.6f}–{:.6f}×".format(
            len(c4), mean_gain(c4), *span(c4, "heldout_gain"), *span(c4, "oversample")
        ),
        "1 run; Δ {:.6f}; zero-loss fraction {:.6f}".format(
            single[2]["heldout_gain"], single[2]["zero_loss_frac"]
        ),
        "1 run; Δ {:+.6f}".format(single[8]["heldout_gain"]),
        "1 run; Δ {:+.6f}".format(single[16]["heldout_gain"]),
        "{} runs; mean Δ {:+.6f}; no groups skipped".format(len(tbb), mean_gain(tbb))
        if all(r["groups_skipped"] == 0 for r in tbb)
        else "groups skipped",
        "{} runs; mean Δ {:+.6f}; {}–{} groups skipped per seed".format(
            len(tbc), mean_gain(tbc), *span(tbc, "groups_skipped")
        ),
        "{} runs; mean Δ {:+.6f}; zero-loss fraction {:.6f}–{:.6f}".format(
            len(hcb), mean_gain(hcb), *span(hcb, "zero_loss_frac")
        ),
        "{} runs; mean Δ {:.6f}; zero-loss fraction {:.6f}".format(
            len(hcc), mean_gain(hcc), max(r["zero_loss_frac"] for r in hcc)
        ),
        "{} runs; mean Δ {:+.6f} (sum) / {:.6f} (mean) / {:+.6f} (surprise)".format(
            len(p4),
            *(
                mean_gain(p4, lambda r, m=m: r["loss_mode"] == m)
                for m in ("sum", "mean", "surprise")
            ),
        ),
        "1 run; Δ {:+.6f}; held-out 20; zero-loss fraction {:.6f}".format(
            co["baseline"]["heldout_gain"], co["baseline"]["zero_loss_frac"]
        ),
        "oversample {:.6f}×".format(co["curriculum"]["oversample_factor"]),
        "({:.6f} = {}/30)".format(camp[0]["heldout_before"], round(camp[0]["heldout_before"] * 30)),
        f"**Table A.6 — Semester-4 opening and null-result runs ({n_runs} runs).**",
    ]


def layer_freeze_rows():
    a, b, c, d = (
        J(f"{RES}/p1_layerfreeze/{n}.json")
        for n in ("result", "scaled_4seed_result", "freeze_flop_result", "emergence_result")
    )
    return [
        "overlap {:.6f}; top-25% concentration {:.6f}".format(
            a["step1_predicts_final_topk_overlap"], a["concentration_top25pct_share"]
        ),
        "overlap {:.6f} ± {:.6f}".format(
            b["step1_predicts_final_topk_overlap_mean"], b["step1_predicts_final_topk_overlap_std"]
        ),
        "concentration {:.6f}".format(b["concentration_top25pct_share_mean"]),
        "at {:.6f}% of the parameter count".format(100 * c["param_ratio_frozen_over_full"]),
        "mean step-1 overlap {:.6f}".format(d["mean_step1"]),
    ]


def a8_rows():
    e1, f1 = J(f"{E1_DIR}/receipt.json"), J(f"{FINISH}/E1/result.json")
    f5, f6 = J(f"{FINISH}/E5/result.json"), J(f"{FINISH}/E6/result.json")
    lb, ad, b, e = J(LABBENCH), J(AGENTDOJO), e11_boot(), e14()
    e4 = J(f"{SMALL}/E4/result.json")["prior_base_run"]
    nat = e1["coverage"]["native_evaluations"]
    t = {lane: table_8c(lane) for lane in ("E2", "E5", "E6", "E9", "E13")}
    return [
        "{}/731 = {:.6f}% pass@1 ({:.6f}% coverage, {}/731 native evals)".format(
            e1["coverage"]["resolved"], e1["score_percent"], pct(nat, 731), nat
        ),
        "{}/{} resolved ({} graded); earlier waves {}/{}".format(
            f1["n_resolved"],
            f1["n_attempted"],
            f1["n_graded"],
            f1["prior_waves_separate_not_pooled"]["resolved"],
            f1["prior_waves_separate_not_pooled"]["attempted"],
        ),
        "{} = {:.6f} task accuracy".format(t["E2"][2], t["E2"][3]),
        "{} = {:.6f} pass^1 ({} infra errors counted as failures)".format(
            t["E5"][2], t["E5"][3], f5["n_infrastructure_error"]
        ),
        "{} = {:.6f}, lower bound; {} ungraded".format(
            t["E6"][2], t["E6"][3], f6["n_ungraded_judge_unavailable"]
        ),
        "**{}/{} COMPLETE**".format(lb["evaluated"], lb["expected_total"]),
        "{} = {:.6f} task success".format(t["E9"][2], t["E9"][3]),
        "{}/97 tasks completed, utility {}/97 = {:.6f}".format(
            ad["completed_episodes"], ad["utility_passes"], ad["score"]
        ),
        "**{n}/{n} = {}/{n} pass@1 ({:.6f}%)**; {}/156 completion + {}/156 spec-to-RTL".format(
            b["cc"] + b["spec"], pct(b["cc"] + b["spec"], 312), b["cc"], b["spec"], n=2 * b["n"]
        ),
        "{}/{} episodes; {:.6f}% progression".format(t["E13"][1], t["E13"][0], t["E13"][2]),
        "{}/{} accepted ({:.6f}%); accuracy {}/{} = {:.6f}% (official scorer {:.6f}% on {} judged rows)".format(
            e["accepted"],
            e["n"],
            pct(e["accepted"], e["n"]),
            e["correct"],
            e["n"],
            pct(e["correct"], e["n"]),
            pct(e["correct"], e["accepted"]),
            e["accepted"],
        ),
        "{n}/{n} trials, mean reward {}".format(e4["mean_reward"], n=e4["n_trials"]),
    ]


def claims_ext():
    yield ("§6 recomputed values", "ch06", CH06_TOKENS, "R", CH06_FILES, ch06_rows)
    recon = f"{RES}/group_size_g4_vs_g32_broader_scale.tsv"
    yield (
        "§6.4.2 reconstruction grid",
        "ch06",
        [
            'a T = 1M comparison of 0.41 against 0.42 with a difference of −0.01, a 97.6% retention ratio and a "generalizes" verdict of yes',
            "at T = 4M, 0.55 against 0.66 with −0.11, 83.3% and no",
            "at T = 16M, 0.63 against 0.84 with −0.21, 75.0% and no",
        ],
        "R",
        [recon],
        lambda: (
            lambda rows: [
                "a T = 1M comparison of {acc_G_a} against {acc_G_b} with a difference of {diff_a_minus_b}, a {r:.6f}% retention ratio "
                'and a "generalizes" verdict of {generalizes_wu_claim}'.format(
                    r=float(rows[0]["retention_pct_of_Gb"]), **rows[0]
                ),
                *(
                    "at T = {T_M_tokens}M, {acc_G_a} against {acc_G_b} with {diff_a_minus_b}, {r:.6f}% and {generalizes_wu_claim}".format(
                        r=float(x["retention_pct_of_Gb"]), **x
                    )
                    for x in rows[1:3]
                ),
            ]
        )(tsv(recon)),
    )
    yield (
        "§6.1 η² sum",
        "ch06",
        ["These four values sum to 1.897"],
        "A",
        [],
        lambda: [f"These four values sum to {0.546 + 0.558 + 0.471 + 0.322:.6f}"],
    )
    for name, tok, src in (
        (
            "registry bootstrap CI",
            "the registry's bootstrap 95% CI is [−0.37, 0.88]",
            f"{RES}/claim_to_run/claim_to_run_table.tsv",
        ),
        ("historical Welch test", "p = 0.7605", "REPRODUCE.md"),
        ("SNR fraction", "about 52% of the √G ideal", f"{SEC}/p3_abstract.tex"),
        ("SNR above flat", "about 48% above a flat baseline", f"{SEC}/group_size_iter27.tex"),
        (
            "superseded ε statement",
            "The earlier ≤0.4% ε-sensitivity statement",
            f"{SEC}/zvf_pipeline_spec.tex",
        ),
        ("reconstruction interval", "[0.873, 1.092]", f"{RES}/findings_ledger.jsonl"),
    ):
        yield (f"§6 {name}", "ch06", [tok], "T", [src], None)
    zvf_files = [f"{RES}/tinker_gsm8k_zvf_s{s}.json" for s in (42, 123, 456)]
    audit_files = [f"{AUDIT}/{a}-seed-{s}.json" for a in ("grpo", "dapo") for s in SEEDS11]
    yield (
        "Table 1.1 recomputed rows",
        "ch01",
        [
            "0.130 / 0.190 / 0.155; pooled 0.158",
            "0.838 / 0.764 / 0.691 / 0.631 at G = 2 / 4 / 8 / 16 (fall of 0.207)",
            "0.982 / 0.988 / 0.990 / 0.978 at G = 2 / 4 / 8 / 16",
            "GRPO 0.990, PPO 0.992; paired Δ = −0.002, p = 0.374",
            "GRPO (G = 8) 0.245, GRPO (G = 2) 0.250, PPO 0.195; GRPO G8 − PPO +0.050 [+0.015, +0.085], "
            "paired-t p = 0.016, exact p = 0.0625; G8 − G2 −0.005 [−0.038, +0.028]",
            "0.987 vs 0.992, p = 0.35; completions 4.7 tokens",
            "0.202 → 0.263 / 0.205 → 0.255",
            "Δ = +0.001, paired-t 95% CI [−0.00632, +0.00832]",
            "0.856 (Tinker, Qwen3-8B-Base) vs 0.050 (TRL",
            "+0.05 in both arms",
        ],
        "R",
        [
            *zvf_files,
            f"{RES}/groupsize_zvf_sweep.json",
            f"{RES}/samestack_ppo_grpo.json",
            f"{RES}/samestack_gsm8k_cot.json",
            f"{RES}/drgrpo_vs_grpo.json",
            f"{RES}/drgrpo_gsm8k_cot.json",
            *audit_files,
            f"{RES}/framework_comparison.json",
            f"{RES}/curriculum_opening/results.json",
        ],
        ch01_rows,
    )
    for name, tok, src in (
        (
            "Phase-1 ZVF",
            "0.72–0.77 (all-correct 0.65–0.71)",
            "platform_tinker/reports/esa_phase1/Phase1_Project_Report_ZVF.tex",
        ),
        (
            "ZVF–outcome ρ",
            "ρ ≈ 0.27, 95% CI [−0.37, 0.88]",
            f"{RES}/claim_to_run/claim_to_run_table.tsv",
        ),
        (
            "pooled ZVF r",
            "r = −0.769",
            "platform_hybrid/paper/neurips_2026_variants/paper_P8_workshop.tex",
        ),
        ("instrumented ZVF r", "r ≈ 0.09", f"{SEC}/appendix_zvf_formalization.tex"),
        (
            "Qwen3-8B pre/post",
            "82.0% → 83.3%, p = 0.26",
            "platform_hybrid/paper/archive/absorbed/U01_main_compendium/main.tex",
        ),
        ("reproduction headline", "last-10 training reward 34.4%, peak 62.5%", "REPRODUCE.md"),
        ("P1 anchor fit", "mean training reward 0.285, peak 0.625", f"{SEC}/scaling_laws.tex"),
        ("pooled Tinker vs TRL", "99.9% vs 73.4%", "LIMITATIONS_AND_IMPACT.md"),
        (
            "layer freeze",
            "+0.0625 (full) vs +0.000 (frozen)",
            f"{RES}/p1_layerfreeze/FREEZE_FINDINGS.md",
        ),
    ):
        yield (f"Table 1.1 {name}", "ch01", [tok], "T", [src], None)
    sweep = J(f"{RES}/groupsize_zvf_sweep.json")
    pp_src = text(f"{M}/modal_samestack_ppo_grpo.py")
    yield (
        "Table 4.1 design numbers",
        "ch04",
        ["(16 × 8 versus 128)", "G ∈ {2, 4, 8, 16}"],
        "R",
        [f"{M}/modal_samestack_ppo_grpo.py", f"{RES}/groupsize_zvf_sweep.json"],
        lambda: [
            (lambda n, g: f"({n // g} × {g} versus {n})")(
                int(rx(r"N_GEN = (\d+)", pp_src, 1)),
                int(rx(r"GROUP = (\d+)", pp_src, 1)),
            ),
            "G ∈ {" + ", ".join(sorted(sweep["summary"], key=int)) + "}",
        ],
    )
    for i, (tail, runner, patterns) in enumerate(code_rows()):
        yield (
            f"Table 4.2 runner {i + 1:02d}",
            "ch04",
            [tail],
            "C",
            [runner],
            lambda p=patterns: [p],
        )
    yield (
        "Table A.1 historical boundary",
        "appA",
        ["commit `21a99ef7` dated 23 April 2026"],
        "R",
        [],
        lambda: [git_tag("capstone-final-2026-04-25")],
    )
    modal = J("modal_results_all.json")

    def a3(exp, seconds=False):
        rows = sorted(modal[exp], key=lambda r: [42, 123, 456, 789, 1024].index(r["seed"]))
        if seconds:
            return "{:.6f}–{:.6f} s".format(
                min(r["elapsed_seconds"] for r in rows), max(r["elapsed_seconds"] for r in rows)
            )
        return "final acc " + " / ".join(f"{r['final_accuracy']:.3f}" for r in rows)

    yield (
        "Table A.3 Modal campaign",
        "appA",
        [
            "final acc 0.735 / 0.810 / 0.620 / 0.740 / 0.765; 136.4–183.7 s",
            "final acc 0.003 / 0.014 / 0.011 / 0.012 / 0.010; `steps_to_95` null for all five",
            "final acc 0.010 / 0.008 / 0.014 / 0.009 / 0.004; 25.4–32.4 s",
            "final acc 0.011 / 0.002 / 0.005 / 0.002 / 0.009; 1.67–3.30 s",
            "| Qwen/Qwen2.5-0.5B | GRPO | — | math | 42, 123, 456, 789, 1024 | 125 |",
            "| not recorded | PPO |",
            "| ~100k env. steps |",
        ],
        "R",
        ["modal_results_all.json"],
        lambda: [
            f"{a3('trl_grpo_math')}; {a3('trl_grpo_math', True)}",
            f"{a3('sb3_ppo_math')}; `steps_to_95` null for all five"
            if all(r.get("steps_to_95") is None for r in modal["sb3_ppo_math"])
            else "steps_to_95 set",
            f"{a3('cleanrl_ppo_math')}; {a3('cleanrl_ppo_math', True)}",
            f"{a3('tianshou_ppo_math')}; {a3('tianshou_ppo_math', True)}",
            "| {} | GRPO | — | math | 42, 123, 456, 789, 1024 | {} |".format(
                *{r["model"] for r in modal["trl_grpo_math"]},
                *{r["train_steps"] for r in modal["trl_grpo_math"]},
            ),
            "| not recorded | PPO |"
            if not any(
                "model" in r or "train_steps" in r
                for k, rs in modal.items()
                if k != "trl_grpo_math"
                for r in rs
            )
            else "PPO model recorded",
            "| ~{}k env. steps |".format(
                *{
                    round(r["learning_curve"][-1][0] / 1e4) * 10
                    for k, rs in modal.items()
                    if k != "trl_grpo_math"
                    for r in rs
                }
            ),
        ],
    )

    def a4(g):
        runs = [r for r in sweep["runs"] if r["group_size"] == g]
        acc = [r["heldout_acc"] for r in runs]
        return "held-out acc {:.6f} ± {:.6f}; mean ZVF {:.6f}".format(
            statistics.mean(acc),
            statistics.stdev(acc) / math.sqrt(len(acc)),
            statistics.mean(r["mean_zvf"] for r in runs),
        )

    yield (
        "Table A.4 group-size sweep",
        "appA",
        [
            f"held-out acc {a} ± {s}; mean ZVF {z}"
            for a, s, z in (
                ("0.9817", "0.0044", "0.8380"),
                ("0.9883", "0.0017", "0.7635"),
                ("0.9900", "0.0029", "0.6906"),
                ("0.9783", "0.0060", "0.6312"),
            )
        ],
        "R",
        [f"{RES}/groupsize_zvf_sweep.json"],
        lambda: [a4(g) for g in (2, 4, 8, 16)],
    )
    yield (
        "Table A.5 evidence tiers",
        "appA",
        ["C = 12, R = 2, X = 4", "eighteen scientific claim rows"],
        "R",
        [f"{RES}/claim_to_run/claim_to_run_table.tsv"],
        lambda: (
            lambda t: [
                f"C = {t['C']}, R = {t['R']}, X = {t['X']}",
                "eighteen scientific claim rows"
                if sum(t.values()) == 18
                else f"{sum(t.values())} rows",
            ]
        )(Counter("C" if r["claim_id"] == "P1-C2" else r["evidence_tier"] for r in claim_rows())),
    )
    fraud = tsv(f"{RES}/quick_20260704/qp8_fraud.tsv")[0]
    yield (
        "Table A.5 recomputed outcomes",
        "appA",
        [
            "G=2 +0.0000, G=4 +0.0000, G=8 −0.0625, G=16 +0.0000",
            "GRPO 0.2017→0.2633 (Δ +0.0617); Dr.GRPO 0.2050→0.2550 (Δ +0.0500)",
            "Final/last-10 training reward 0.050→0.856",
            "44/48 entries pass",
            "AUC 0.7955 on the 10,000-row",
            "| P7-C1 | 368 —",
        ],
        "R",
        [
            f"{RES}/groupsize_zvf/results.json",
            f"{RES}/drgrpo_gsm8k_cot.json",
            f"{RES}/framework_comparison.json",
            f"{RES}/claim_to_run/claim_to_run_table.tsv",
            f"{RES}/quick_20260704/qp8_fraud.tsv",
        ],
        lambda: (
            lambda gz, dc, fw, cr: [
                ", ".join(
                    "G={} {:+.4f}".format(g, mean_gain(gz, lambda r, g=g: r["G"] == g)).replace(
                        "-", "−"
                    )
                    for g in (2, 4, 8, 16)
                ),
                "GRPO {:.4f}→{:.4f} (Δ {:+.4f}); Dr.GRPO {:.4f}→{:.4f} (Δ {:+.4f})".format(
                    *(
                        dc["summary"][a][k]
                        for a in ("grpo", "dr_grpo")
                        for k in ("heldout_pre_mean", "heldout_post_mean", "delta_mean")
                    )
                ),
                "Final/last-10 training reward {:.6f}→{:.6f}".format(
                    fw["TRL"]["last10_avg"], fw["Tinker"]["last10_avg"]
                ),
                rx(r"\d+/\d+ entries pass", cr["P6-C1"]["heldout_metric"]),
                "AUC {} on the {:,}-row".format(fraud["auc"], int(fraud["n"])),
                "| P7-C1 | {} —".format(len(cr["P7-C1"]["run_ids"].split(";"))),
            ]
        )(
            J(f"{RES}/groupsize_zvf/results.json"),
            J(f"{RES}/drgrpo_gsm8k_cot.json"),
            {x["framework"]: x for x in J(f"{RES}/framework_comparison.json")["frameworks"]},
            {r["claim_id"]: r for r in claim_rows()},
        ),
    )
    yield (
        "Table A.5 P7-C1 roster split",
        "appA",
        [
            "Qwen3-4B-Instruct-2507 (113), Llama-3.2-3B (112), Llama-3.2-1B (107), 3 further models (36)"
        ],
        "A",
        [],
        lambda: [None],
    )
    yield (
        "Table A.5 transcribed outcomes",
        "appA",
        [
            "Spearman 0.56, point-biserial 0.62 (n = 23)",
            "Peak 0.875, last-10 0.1625, zero-reward step fraction 0.55 (11/20)",
            "Pearson r = +0.71",
            "Adaptive-G held-out Δ +0.575; mean ZVF 0.23; 186 rollouts",
            "AUC 0.482675, accuracy 0.792 (n = 500)",
        ],
        "T",
        [f"{RES}/claim_to_run/claim_to_run_table.md"],
        None,
    )
    yield (
        "Table A.5 P11 interval",
        "appA",
        ["percentile-bootstrap CI [−0.0045, +0.00675]", "MDE₈₀ = 0.01012"],
        "T",
        ["zvf-program/audit/STATISTICAL_REANALYSIS.md"],
        None,
    )
    yield (
        "Table A.5 P11 diagnostics",
        "appA",
        [
            "paired-t 95% CI [−0.0063, +0.0083], n = 8 seeds",
            "DAPO's mean ZVF is 0.000 against GRPO's 0.693",
            "3.61×",
            "(1,734 versus 480)",
            "1.44× the wall clock",
        ],
        "R",
        audit_files,
        lambda: (
            lambda p: [
                "paired-t 95% CI [{:+.6f}, {:+.6f}], n = 8 seeds".format(p["lo"], p["hi"]),
                "DAPO's mean ZVF is {:.6f} against GRPO's {:.6f}".format(*p["zvf"]),
                "{:.6f}×".format(p["roll"][0] / p["roll"][1]),
                "({:.6f} versus {:.6f})".format(*p["roll"]),
                "{:.6f}× the wall clock".format(p["wall"]),
            ]
        )(p11()),
    )
    yield (
        "Table A.6 opening runs",
        "appA",
        [
            "6 runs; mean Δ +0.0111 (0.0000–0.0333)",
            "6 runs; mean Δ +0.0167 (−0.0667 to +0.0667); oversample 3.3–7.2×",
            "1 run; Δ 0.0000; zero-loss fraction 0.7",
            "1 run; Δ −0.1333",
            "1 run; Δ +0.0333",
            "3 runs; mean Δ +0.0278; no groups skipped",
            "3 runs; mean Δ +0.0278; 11–20 groups skipped per seed",
            "3 runs; mean Δ +0.0278; zero-loss fraction 0.1–0.2",
            "3 runs; mean Δ 0.0000; zero-loss fraction 0.0",
            "6 runs; mean Δ +0.1250 (sum) / 0.0000 (mean) / +0.0417 (surprise)",
            "1 run; Δ +0.0500; held-out 20; zero-loss fraction 0.50",
            "oversample 4.81×",
            "(0.8667 = 26/30)",
            "**Table A.6 — Semester-4 opening and null-result runs (46 runs).**",
        ],
        "R",
        [
            f"{RES}/{n}/results.json"
            for n in (
                "campaign",
                "token_budget",
                "hard_curriculum",
                "p4_surprise",
                "curriculum_opening",
            )
        ],
        a6_rows,
    )
    lf = [
        f"{RES}/p1_layerfreeze/{n}.json"
        for n in ("result", "scaled_4seed_result", "freeze_flop_result", "emergence_result")
    ]
    yield (
        "Table A.6 layer-freeze runs",
        "appA",
        [
            "overlap 1.0; top-25% concentration 0.476",
            "overlap 0.0833 ± 0.0481",
            "concentration 0.3908",
            "at 60.7% of the parameter count",
            "mean step-1 overlap 0.643",
        ],
        "S",
        lf,
        layer_freeze_rows,
    )
    yield (
        "Table A.7 same-stack rerun runs",
        "appA",
        [
            "post 0.230–0.255, mean 0.245",
            "post 0.230–0.280, mean 0.250",
            "post 0.160–0.215, mean 0.195",
        ],
        "R",
        [f"{RES}/samestack_gsm8k_cot.json"],
        lambda: [
            (
                lambda xs: "post {:.6f}–{:.6f}, mean {:.6f}".format(
                    min(xs), max(xs), statistics.mean(xs)
                )
            )(
                [
                    r["heldout_post_acc"]
                    for r in J(f"{RES}/samestack_gsm8k_cot.json")["runs"]
                    if r["arm"] == a
                ]
            )
            for a in ("grpo_g8", "grpo_g2", "ppo")
        ],
    )
    yield (
        "Table A.8 campaign lanes",
        "appA",
        [
            "2/731 = 0.274% pass@1 (97.54% coverage, 713/731 native evals)",
            "1/190 resolved (57 graded); earlier waves 4/110",
            "27/45 = 0.600 task accuracy",
            "8/97 = 0.082 pass^1 (25 infra errors counted as failures)",
            "90/812 = 0.111, lower bound; 108 ungraded",
            "**1967/1967 COMPLETE**",
            "10/34 = 0.294 task success",
            "97/97 tasks completed, utility 88/97 = 0.907",
            "**312/312 = 129/312 pass@1 (41.35%)**; 67/156 completion + 62/156 spec-to-RTL",
            "255/255 episodes; 26.1% progression",
            "4426/4428 accepted (99.95%); accuracy 2271/4428 = 51.29% (official scorer 51.31% on 4426 judged rows)",
            "100/100 trials, mean reward 0.0",
        ],
        "R",
        [
            f"{E1_DIR}/receipt.json",
            *(f"{FINISH}/{x}/result.json" for x in ("E1", "E2", "E5", "E6", "E9", "E13")),
            LABBENCH,
            AGENTDOJO,
            E11_BOOT,
            E14_DISP,
            f"{SMALL}/E4/result.json",
        ],
        a8_rows,
    )
    yield (
        "Table A.8 transcribed partials",
        "appA",
        [
            "1/17 tasks; replay normalized 0.8628",
            "1/100 tasks, recovery 0.3115",
            "7/480 native-scored; prefix mean 0.050505",
            "40/75 natively graded",
        ],
        "T",
        [CAMPAIGN_MD],
        None,
    )
    yield (
        "Table A.8 E9 coverage",
        "appA",
        ["40/75 natively graded (53.33%)"],
        "A",
        [],
        lambda: [f"40/75 natively graded ({pct(40, 75):.2f}%)"],
    )
    yield ("Table A.8 E7 attempts", "appA", ["1/46 attempted"], "T", [FINAL_MD], None)
    yield (
        "Table A.10 corpora",
        "appA",
        ["| 79 |", "| 790 |", "| 98 |", "| 15 |"],
        "R",
        [
            "platform_hybrid/paper/neurips_2026_variants/paper_P8_workshop.tex",
            f"{SEC}/p5_stack.tex",
            f"{STATS}/m18_mantel_summary.json",
            f"{RES}/samestack_gsm8k_cot.json",
        ],
        lambda: [
            "| 79 |"
            if in_source(
                "79", text("platform_hybrid/paper/neurips_2026_variants/paper_P8_workshop.tex")
            )
            else "absent",
            "| 790 |" if in_source("790", text(f"{SEC}/p5_stack.tex")) else "absent",
            f"| {J(f'{STATS}/m18_mantel_summary.json')['n_cells']} |",
            f"| {len(J(f'{RES}/samestack_gsm8k_cot.json')['runs'])} |",
        ],
    )
    yield (
        "Table A.10 named-run total",
        "appA",
        ["This appendix names 105 runs individually"],
        "A",
        [],
        lambda: [f"This appendix names {4 + 20 + 12 + 46 + 23} runs individually"],
    )
    yield ("Table A.10 P1 roster", "appA", ["| 70+ |"], "T", [f"{SEC}/p1_abstract.tex"], None)


# ---------------------------------------------------------------- chapter 6 recomputations
def tost_p(diffs, margin):
    from scipy import stats

    n, m, sd = len(diffs), statistics.mean(diffs), statistics.stdev(diffs)
    se = sd / math.sqrt(n)
    return max(stats.t.sf((m + margin) / se, n - 1), stats.t.cdf((m - margin) / se, n - 1))


def mde_d(n1, n2, power=0.8):
    from scipy import stats

    df = n1 + n2 - 2
    scale = math.sqrt(n1 * n2 / (n1 + n2))
    tcrit = stats.t.ppf(0.975, df)
    lo, hi = 0.01, 20.0
    for _ in range(100):
        d = (lo + hi) / 2
        if stats.nct.sf(tcrit, df, d * scale) < power:
            lo = d
        else:
            hi = d
    return d


def ch06_rows():
    from scipy import stats

    fits = {r["model"]: r for r in tsv(f"{RES}/scaling_law_fits.tsv")}
    pp = J(f"{RES}/samestack_ppo_grpo.json")["runs"]
    by = {(r["algo"], r["seed"]): r["heldout_acc"] for r in pp}
    seeds = [42, 123, 456, 789, 1024]
    diffs = [by["grpo", s] - by["ppo", s] for s in seeds]
    tol = J(f"{RES}/pcd_vs_zvf_tolerance_analysis.json")
    sw = J(f"{RES}/groupsize_zvf_sweep.json")
    acc = {g: [r["heldout_acc"] for r in sw["runs"] if r["group_size"] == g] for g in (2, 4, 8, 16)}
    f_stat, f_p = stats.f_oneway(*acc.values())
    dpo = tsv(f"{RES}/group_size_effect_dpo_check.tsv")[0]
    fac = json.dumps(J(f"{RES}/berkeley/unpacking_dpo_ppo_factorization.json"))
    fac_val = {k: float(v) for k, v in re.findall(r'"(eta2|cohen_d_2v16)": (-?[\d.]+)', fac)}
    tb = J(f"{RES}/token_budget/results.json")
    lf1, lf2, lf4 = (
        J(f"{RES}/p1_layerfreeze/{n}.json")
        for n in ("result", "scaled_result", "scaled_4seed_result")
    )
    dv = J(f"{RES}/drgrpo_vs_grpo.json")
    dc = J(f"{RES}/drgrpo_gsm8k_cot.json")["summary"]
    full = J(f"{RES}/drgrpo_gsm8k_cot_full.json")["runs"]
    run = {(r["algo"], r["seed"]): r for r in full}

    def seed_deltas(algo):
        return ", ".join(
            "{:+.6f}".format(run[algo, s]["heldout_post_acc"] - run[algo, s]["heldout_pre_acc"])
            for s in (42, 123, 456)
        )

    def seed_mcnemar(algo):
        return ", ".join(
            "{:.6f}".format(
                mcnemar(list(zip(run[algo, s]["post_correct"], run[algo, s]["pre_correct"])))[2]
            )
            for s in (42, 123, 456)
        )

    forms = sorted({tuple(int(x) for x in r["pre_correct"]) for r in full}, key=sum)
    differ = sum(len(set(col)) > 1 for col in zip(*forms))
    sd = {g: statistics.stdev(v) for g, v in acc.items()}
    se = {g: sd[g] / math.sqrt(3) for g in acc}
    return [
        "Qwen3.5-4B fits a mean reward of {:.6f} with a peak of {:.6f} and λ pinned at the bound of {:.6f}".format(
            *(float(fits["Qwen3.5-4B"][k]) for k in ("mean_reward", "peak", "lambda"))
        ),
        "Qwen3-8B fits {:.6f} with a peak of {:.6f}".format(
            *(float(fits["Qwen3-8B"][k]) for k in ("mean_reward", "peak"))
        ),
        "Llama-3.1-8B-Instruct fits {:.6f} with a peak of {:.6f}".format(
            *(float(fits["Llama-3.1-8B-Instruct"][k]) for k in ("mean_reward", "peak"))
        ),
        "DeepSeek-V3.1 fits {:.6f} with a peak of {:.6f}".format(
            *(float(fits["DeepSeek-V3.1"][k]) for k in ("mean_reward", "peak"))
        ),
        "evaluation accuracy is {:.6f} for GRPO and {:.6f} for PPO".format(
            statistics.mean(by["grpo", s] for s in seeds),
            statistics.mean(by["ppo", s] for s in seeds),
        ),
        "The five per-seed differences are "
        + ", ".join("{:+.6f}".format(d) for d in diffs[:-1])
        + " and {:+.6f}".format(diffs[-1]),
        "At a {:.6f}–{:.6f} ceiling".format(min(by.values()), max(by.values())),
        "A paired TOST with a ±0.005 margin gives p = {:.6f}".format(tost_p(diffs, 0.005)),
        "it is at ±0.010 (p = {:.6f})".format(tost_p(diffs, 0.010)),
        "PCD changes by {:+.6f} × 10⁻⁷".format(tol["pcd_delta"] * 1e7),
        "sample variance at most {:.6f} × 10⁻⁹".format(
            8 * (0.5 * tol["jitter_amplitude"]) ** 2 / 7 * 1e9
        ),
        "from {:.6f} to {:.6f}".format(
            1 - sw["summary"]["2"]["mean_zvf"], 1 - sw["summary"]["16"]["mean_zvf"]
        ),
        "Spearman ρ = 0.56 (Fisher 95% CI [{:.6f}, {:.6f}], n = 23 pooled cells)".format(
            *fisher_ci(0.56, 23)
        ),
        "the Fisher CI is [{:.6f}, {:.6f}], p = {:.6f}, n = 23".format(
            *fisher_ci(0.27, 23), 2 * stats.t.sf(0.27 * math.sqrt(21 / (1 - 0.27**2)), 21)
        ),
        "Values are mean ± SE over n = 3 seeds: "
        + ", ".join(
            "{:.6f} ± {:.6f} (SD {:.6f}) at G = {}".format(statistics.mean(acc[g]), se[g], sd[g], g)
            for g in (8, 2)
        )
        + ", {:.6f} ± {:.6f} (SD {:.6f}) at G = 4 and {:.6f} ± {:.6f} (SD {:.6f}) at G = 16".format(
            statistics.mean(acc[4]), se[4], sd[4], statistics.mean(acc[16]), se[16], sd[16]
        ),
        "F(3, 8) = {:.6f}, p = {:.6f}".format(f_stat, f_p),
        "d = {:.6f}".format(mde_d(3, 3)),
        "records G = 2 and G = 16 at {} and {}, a difference of +{}".format(
            dpo["mean_a"], dpo["mean_b"], dpo["diff_a_minus_b"]
        ),
        "G = 2 retains {:.6f}% of G = 16".format(float(dpo["retention_pct_of_G16"])),
        "η² = {:.6f}, from the same five-seed".format(fac_val["eta2"]),
        "Cohen's d = {:.6f}".format(fac_val["cohen_d_2v16"]),
        "ties the baseline at {:+.6f} against {:+.6f}".format(
            mean_gain([r for r in tb if r["mode"] == "curriculum"]),
            mean_gain([r for r in tb if r["mode"] == "baseline"]),
        ),
        'a single-seed G = 4 "win" of {:+.6f}'.format(
            next(
                r["heldout_gain"]
                for r in J(f"{RES}/p3_groupsize/sweep_results.json")
                if r["G"] == 4
            )
        ),
        "a per-step overlap of {:.6f}".format(lf1["step1_predicts_final_topk_overlap"]),
        "but only {:.6f}".format(lf2["step1_predicts_final_topk_overlap_mean"]),
        "still holds at approximately {:.6f}".format(lf2["concentration_top25pct_share_mean"]),
        "mean overlap of {:.6f} ± {:.6f} (mean ± SD, n = 4 seeds)".format(
            lf4["step1_predicts_final_topk_overlap_mean"],
            lf4["step1_predicts_final_topk_overlap_std"],
        ),
        "evaluation accuracy is {:.6f} for GRPO and {:.6f} for Dr. GRPO (paired p = {:.6f}, five seeds), and completions average {:.6f} tokens".format(
            dv["summary"]["grpo"]["heldout_mean"],
            dv["summary"]["dr_grpo"]["heldout_mean"],
            next(v for k, v in dv["paired_drgrpo_vs_grpo"].items() if k.startswith("p_two")),
            dv["summary"]["grpo"]["mean_comp_len"],
        ),
        "length drifts from {:.6f} to {:.6f} tokens".format(
            statistics.mean(dc[a]["comp_len_first5"] for a in ("grpo", "dr_grpo")),
            statistics.mean(dc[a]["comp_len_last5"] for a in ("grpo", "dr_grpo")),
        ),
        "{:+.6f} percentage points for GRPO and {:+.6f} for Dr. GRPO".format(
            100 * dc["grpo"]["delta_mean"], 100 * dc["dr_grpo"]["delta_mean"]
        ),
        "(deltas {})".format(seed_deltas("grpo")),
        "Dr. GRPO ({})".format(seed_deltas("dr_grpo")),
        "per-seed McNemar exact p-values are {} for GRPO".format(seed_mcnemar("grpo")).replace(
            ", ", ", ", 1
        ),
        "and {} for Dr. GRPO; no pooled".format(seed_mcnemar("dr_grpo")),
        "two distinct forms, {}, which disagree on {} of the 200 items".format(
            " and ".join(f"{sum(f)}/200" for f in forms), differ
        ),
        "About {:.6f}% of items".format(pct(differ, 200)),
        "mean difference {:+.6f}".format(
            J(f"{RES}/berkeley/adding_error_bars_summary.json")["seven_headline_audit"][6][
                "delta_mean"
            ]
        ),
    ]


CH06_TOKENS = [
    "Qwen3.5-4B fits a mean reward of 0.817 with a peak of 1.000 and λ pinned at the bound of 10.0",
    "Qwen3-8B fits 0.285 with a peak of 0.625",
    "Llama-3.1-8B-Instruct fits 0.869 with a peak of 1.000",
    "DeepSeek-V3.1 fits 0.844 with a peak of 1.000",
    "evaluation accuracy is 0.990 for GRPO and 0.992 for PPO",
    "The five per-seed differences are −0.005, +0.005, −0.005, 0.000 and −0.005",
    "At a 0.98–1.00 ceiling",
    "A paired TOST with a ±0.005 margin gives p = 0.104",
    "it is at ±0.010 (p = 0.008)",
    "PCD changes by +2.984 × 10⁻⁷",
    "sample variance at most 2.86 × 10⁻⁹",
    "from 0.162 to 0.369",
    "Spearman ρ = 0.56 (Fisher 95% CI [0.19, 0.79], n = 23 pooled cells)",
    "the Fisher CI is [−0.16, 0.61], p = 0.21, n = 23",
    "Values are mean ± SE over n = 3 seeds: 0.9900 ± 0.0029 (SD 0.0050) at G = 8, 0.9817 ± 0.0044 (SD 0.0076) at G = 2, "
    "0.9883 ± 0.0017 (SD 0.0029) at G = 4 and 0.9783 ± 0.0060 (SD 0.0104) at G = 16",
    "F(3, 8) = 1.82, p = 0.22",
    "d = 3.07",
    "records G = 2 and G = 16 at 0.9817 and 0.9783, a difference of +0.0033",
    "G = 2 retains 100.3% of G = 16",
    "η² = 0.023, from the same five-seed",
    "Cohen's d = −1.47",
    "ties the baseline at +0.028 against +0.028",
    'a single-seed G = 4 "win" of +0.125',
    "a per-step overlap of 1.0",
    "but only 0.11",
    "still holds at approximately 0.39",
    "mean overlap of 0.083 ± 0.048 (mean ± SD, n = 4 seeds)",
    "evaluation accuracy is 0.987 for GRPO and 0.992 for Dr. GRPO (paired p = 0.35, five seeds), and completions average 4.7 tokens",
    "length drifts from 194.4 to 183.5 tokens",
    "+6.2 percentage points for GRPO and +5.0 for Dr. GRPO",
    "(deltas +0.055, +0.060, +0.070)",
    "Dr. GRPO (+0.045, +0.060, +0.045)",
    "per-seed McNemar exact p-values are 0.071, 0.036 and 0.016 for GRPO",
    "and 0.122, 0.017 and 0.122 for Dr. GRPO; no pooled",
    "two distinct forms, 40/200 and 41/200, which disagree on 7 of the 200 items",
    "About 3.5% of items",
    "mean difference +0.020",
]
CH06_FILES = [
    f"{RES}/scaling_law_fits.tsv",
    f"{RES}/samestack_ppo_grpo.json",
    f"{RES}/pcd_vs_zvf_tolerance_analysis.json",
    f"{RES}/groupsize_zvf_sweep.json",
    f"{RES}/group_size_effect_dpo_check.tsv",
    f"{RES}/berkeley/unpacking_dpo_ppo_factorization.json",
    f"{RES}/token_budget/results.json",
    f"{RES}/p3_groupsize/sweep_results.json",
    f"{RES}/p1_layerfreeze/result.json",
    f"{RES}/p1_layerfreeze/scaled_result.json",
    f"{RES}/p1_layerfreeze/scaled_4seed_result.json",
    f"{RES}/drgrpo_vs_grpo.json",
    f"{RES}/drgrpo_gsm8k_cot.json",
    f"{RES}/drgrpo_gsm8k_cot_full.json",
    f"{RES}/berkeley/adding_error_bars_summary.json",
]
# Values the prose states as definitions or thresholds, not as results.
PROSE_DEFINITIONS = {"95%", "0.05", "65%", "90%", "0.90", "0.5"}


# ---------------------------------------------------------------- prose sweep
CITE = re.compile(r"\((?:sources?|recomputed (?:by|from)): ([^)]*)\)")
PROSE_NUM = re.compile(r"(?<![\w.§-])(?:[-−+]?\d[\d,]*\.\d+%?|\d+(?:\.\d+)?%)(?![A-Za-z\d])")
SKIP_BEFORE = re.compile(
    r"(?:§|Table |Figure |Appendix |Chapter |Exhibit |[Ii]tem |iter(?:ation)?[- ]?)$"
)


def prose_claims(ch):
    """Each number in a cited span of the chapter's prose must occur in a file that span cites."""
    section = ""
    for para in text(f"{THESIS}/{CHAPTERS[ch]}").split("\n\n"):
        head = re.match(r"#{2,4} (\d+(?:\.\d+)+)", para)
        if head:
            section = head[1]
            continue
        if para.lstrip().startswith(("|", ":", "#", "**Table")):
            continue
        start = 0
        for m in CITE.finditer(para):
            span, start = para[start : m.start()], m.end()
            paths = [p.strip(" `").split(" ")[0] for p in re.split(r"[;,]", m[1])]
            paths += [
                re.sub(r"^results/", f"{RES}/", re.sub(r"^sections/", f"{SEC}/", p))
                for p in re.findall(r"`([\w./-]+/[\w.-]+\.\w+)`", span)
            ]
            paths = list(dict.fromkeys(p for p in paths if "/" in p and (ROOT / p).is_file()))
            if not paths:
                continue
            if "corrected" in span or "Table E.1" in span:
                paths.append(f"{THESIS}/ch_back_errata.md")
            tokens = [
                n[0]
                for n in PROSE_NUM.finditer(span)
                if not SKIP_BEFORE.search(span[max(0, n.start() - 12) : n.start()])
            ]
            if tokens:
                yield (f"§{section} prose", ch, list(dict.fromkeys(tokens)), "P", paths, None)


# ---------------------------------------------------------------- evaluation
def norm_num(token):
    s = token.strip().replace("−", "-").replace(",", "").rstrip("%").lstrip("+")
    return float(s), len(s.split(".")[1]) if "." in s else 0


NUM = re.compile(r"[-−+]?\d[\d,]*(?:\.\d+)?")


def numbers_agree(token, computed):
    """Same numbers in the same order, each within half a unit of the token's last printed digit."""
    want, got = NUM.findall(token), NUM.findall(computed)
    if len(want) != len(got) or not want:
        return False
    for w, g in zip(want, got):
        wv, digits = norm_num(w)
        if abs(norm_num(g)[0] - wv) > 0.5 * 10**-digits + 1e-6:
            return False
    return True


def in_source(token, source):
    """Every number of the token occurs in the source, ignoring TeX/markdown decoration."""
    flat = re.sub(r"\\[,;: ]", " ", source)
    flat = re.sub(r"[\\$~{}]", "", flat).replace("−", "-")
    flat = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", flat)
    return all(
        re.search(
            r"(?<![\d.])"
            + re.escape(n.replace("−", "-").replace(",", "").lstrip("+"))
            + (r"0*(?![\d])" if "." in n else r"(?![\d])"),
            flat,
        )
        for n in NUM.findall(token)
    )


def evaluate():
    chapters = {k: text(f"{THESIS}/{v}") for k, v in CHAPTERS.items()}
    groups = []
    verified: dict[str, list[str]] = {}
    for group, ch, tokens, method, artifacts, compute in itertools.chain(
        claims(), claims_ext(), prose_claims("ch06")
    ):
        missing = [
            a
            for a in artifacts
            if not ((ROOT / a).is_file() or (any(c in a for c in "*?[") and expand(a)))
        ]
        record = {
            "group": group,
            "chapter": CHAPTERS[ch],
            "method": method,
            "artifacts": [
                {"path": a, "sha256": None if a in missing else sha256(a)} for a in artifacts
            ],
            "tokens": [],
            "status": "PASS",
        }
        if group in NOTES:
            record["note"] = NOTES[group]
        values = None
        if method in ("R", "S", "A", "C") and not missing:
            try:
                values = compute()
            except (KeyError, ValueError, TypeError, IndexError, ZeroDivisionError, OSError) as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
        for i, tok in enumerate(tokens):
            row = {"token": tok, "in_chapter": tok in chapters[ch]}
            if method == "W":
                row["result"] = "NOT_CHECKED"
            elif missing:
                row["result"] = "ARTIFACT_MISSING"
            elif method in ("T", "P"):
                row["result"] = (
                    "PASS" if any(in_source(tok, text(a)) for a in artifacts) else "FAIL"
                )
                if method == "P" and row["result"] == "FAIL":
                    if any(tok in v for v in verified.get(ch, [])):
                        row["result"] = "PASS"
                        row["note"] = "verified by an explicit recomputation in this chapter"
                    elif tok in PROSE_DEFINITIONS:
                        row["result"] = "CONTEXT"
                    else:
                        row["result"] = "UNBOUND"
            elif method == "C" and values is not None:
                row["patterns"] = values[i]
                row["result"] = (
                    "PASS"
                    if all(any(re.search(x, text(a)) for a in artifacts) for x in values[i])
                    else "FAIL"
                )
            elif values is None:
                row["result"] = "FAIL"
            else:
                v = values[i]
                row["recomputed"] = v if isinstance(v, (str, int, float)) or v is None else str(v)
                if v is None:
                    row["result"] = "CONTEXT"
                elif isinstance(v, str):
                    row["result"] = "PASS" if v == tok or numbers_agree(tok, v) else "FAIL"
                else:
                    want, digits = norm_num(tok)
                    row["result"] = "PASS" if abs(v - want) <= 0.5 * 10**-digits + 1e-9 else "FAIL"
            if not row["in_chapter"]:
                row["result"] = "FAIL"
                row["note"] = "token not found verbatim in chapter"
            record["tokens"].append(row)
            if row["result"] == "PASS" and method != "P":
                verified.setdefault(ch, []).append(tok)
        results = {r["result"] for r in record["tokens"]}
        if "FAIL" in results:
            record["status"] = "FAIL"
        elif "UNBOUND" in results:
            record["status"] = "UNBOUND"
        elif "ARTIFACT_MISSING" in results:
            record["status"] = "ARTIFACT_MISSING"
        elif method == "W":
            record["status"] = "NOT_CHECKED"
        groups.append(record)
    return groups


def short_path(paths):
    names = sorted({Path(p).name for p in paths})
    if len(paths) == 1:
        return names[0]
    return f"{len(paths)} × {names[0]}" if len(names) == 1 else f"{len(paths)} files"


def digest(artifacts):
    """One hash for a group: SHA-256 itself, or SHA-256 over 'sha  path' lines."""
    shas = [a["sha256"] for a in artifacts]
    if not artifacts or None in shas:
        return "—"
    if len(artifacts) == 1:
        return shas[0][:12]
    lines = "".join(
        f"{a['sha256']}  {a['path']}\n" for a in sorted(artifacts, key=lambda a: a["path"])
    )
    return "Σ" + hashlib.sha256(lines.encode()).hexdigest()[:11]


def location(group):
    """Appendix row for a claim group: its table, else its major section."""
    m = re.match(r"(Table [A-Z0-9]+\.[A-Z0-9]+)", group)
    if m:
        return m[1]
    if group.startswith("§6 "):
        return "§6 recomputations"
    m = re.match(r"§(\d+\.\d+)", group)
    return f"§{m[1]}" if m else group


def appendix(groups, report_name):
    checked = [g for g in groups if g["method"] != "W"]
    tokens = [r for g in checked for r in g["tokens"]]
    n_pass = sum(r["result"] in ("PASS", "CONTEXT") for r in tokens)
    n_ctx = sum(r["result"] == "CONTEXT" for r in tokens)
    n_w = sum(len(g["tokens"]) for g in groups if g["method"] == "W")
    rows: dict[str, list] = {}
    order = list(CHAPTERS.values())
    for g in sorted(groups, key=lambda g: order.index(g["chapter"])):
        rows.setdefault(location(g["group"]), []).append(g)
    out = [
        "# Appendix J. Number-to-Artifact Provenance",
        "",
        f"Each row binds the values printed at one location to the bytes they were checked against. "
        f"`tools/audit_thesis_numbers.py` regenerates this table and `{report_name}`, which lists every value, "
        f"its check and each file's full path and SHA-256. A value passes only if it appears verbatim in its "
        f"chapter and agrees at its printed precision. Methods: R recomputed from rows (counts, means, SDs, "
        f"Wilson, Fisher, t, TOST, ANOVA, McNemar, sign-flip, cluster bootstrap, `git apply --stat`); "
        f"S stored summary field; C hyperparameter matched in the runner code; T verbatim in the cited source report; "
        f"P prose sweep, every decimal and percentage in a cited span found in a file that span cites or verified by "
        f"an R row ({n_ctx} values stated as thresholds or definitions are exempt); A arithmetic from printed "
        f"counts, artifact absent from this checkout; W withheld source, not checked. Σ marks the SHA-256 of a "
        f"group's sorted `sha256  path` lines. Hashes bind the author's checkout; Appendix I states which files "
        f"the public package includes.",
        "",
        "\\newpage",  # keep the table on one page
        "",
        "| Location | Values | M | Artifacts | SHA-256 (prefix) | Result |",
        # dash counts set pandoc column widths
        "|----------------------|-----:|:----:|------------------------------------------|---------------|-------------|",
    ]

    def position(loc):
        body = text(f"{THESIS}/{rows[loc][0]['chapter']}")
        if loc.startswith("Table"):
            first = next((r["token"] for g in rows[loc] for r in g["tokens"]), "")
            hit = body.find(f"**{loc} ")
            hit = hit if hit >= 0 else body.find(first)
        else:
            hit = body.find(f"## {loc.lstrip('§').split(' ')[0]} ") if " " not in loc else -1
        return order.index(rows[loc][0]["chapter"]), hit if hit >= 0 else len(body)

    for loc in sorted(rows, key=position):
        gs = rows[loc]
        arts = list({a["path"]: a for g in gs for a in g["artifacts"]}.values())
        toks = [r for g in gs for r in g["tokens"]]
        bad = sum(r["result"] in ("FAIL", "ARTIFACT_MISSING") for r in toks)
        unbound = sum(r["result"] == "UNBOUND" for r in toks)
        skipped = sum(r["result"] == "NOT_CHECKED" for r in toks)
        parts = [f"{bad} disagree"] * bool(bad) + [f"{unbound} unbound"] * bool(unbound)
        parts += [f"{skipped} not checked"] * bool(skipped)
        methods = "".join(sorted({g["method"] for g in gs}, key="RSCTPAW".index))
        artifact = f"`{short_path([a['path'] for a in arts])}`" if arts else "—"
        out.append(
            f"| {loc} | {len(toks)} | {methods} | {artifact} | `{digest(arts)}` | "
            f"{'; '.join(parts) or 'agree'} |"
        )
    out += [
        "",
        f": Number-to-artifact provenance for Chapters 1, 4 and 6–9 and Appendices A and F. {n_pass} of "
        f"{len(tokens)} checked values agree"
        + (
            f"; {n_w} values rest on withheld sources and are not checked."
            if n_w
            else "; no value is left unchecked."
        ),
        "",
    ]
    return "\n".join(out)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--json", type=Path, default=ROOT / THESIS / "provenance_audit.json")
    parser.add_argument("--appendix", type=Path, default=ROOT / THESIS / "ch_back_provenance.md")
    parser.add_argument(
        "--check", action="store_true", help="exit 1 if any checked value disagrees"
    )
    args = parser.parse_args(argv)
    groups = evaluate()
    report = {
        "schema_version": "thesis-number-provenance-v1",
        "status": "FAIL"
        if any(g["status"] in ("FAIL", "ARTIFACT_MISSING") for g in groups)
        else "PASS",
        "scope": "Printed values in the results chapters versus local artifacts; no model, scorer or provider run.",
        "groups": groups,
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    for pattern, replacement in REDACT.items():
        rendered = re.sub(pattern, replacement, rendered)
    args.json.write_text(rendered, encoding="utf-8")
    args.appendix.write_text(appendix(groups, args.json.name), encoding="utf-8")
    for g in groups:
        bad = [r for r in g["tokens"] if r["result"] not in ("PASS", "CONTEXT", "NOT_CHECKED")]
        print(f"{g['status']:<16} {g['method']} {g['group']}")
        for r in bad:
            print(
                f"    {r['result']}: {r['token']!r} -> {r.get('recomputed')!r} {r.get('note', '')}"
            )
    print(f"overall: {report['status']}")
    return 1 if args.check and report["status"] != "PASS" else 0


if __name__ == "__main__":
    raise SystemExit(main())
