#!/usr/bin/env python3
"""Jev use-case layer for the Tinker RL Lab campaign.

Implements the eight builder patterns (2026-09-19 briefing) as concrete,
receipted primitives for THIS project.  Every judgment:

- runs only when the service passes `healthcheck()` (fast path),
- abstains inside a probability band instead of trusting a single threshold,
- falls back to the pre-Jev behaviour (route to the orchestrator/human) on
  abstain or outage, so the worst case is last week's behaviour,
- writes a receipt binding state, question, answer, and routing decision.

Patterns -> primitives here:

  U1 fast-path/fallback  -> `judged()`                      (Recruitly)
  U2 abstain band        -> `AbstainPolicy` (noul bands)     (Spanish legal)
  U3 rerank & fuse       -> `rerank_fuse()` RRF              (legal team)
  U4 tiered eval judge   -> `eval_judge()`                   (Braintrust)
  U5 harness reflexes    -> `safety_gate()`                  (AutoMode style)
  U6 action selection    -> `next_action()`                  (Browser Use)
  U7 bulk archive mining -> `mine_ledgers()`                 (email triage)
  U8 semantic shell filt -> see jgrep.py (exit-code CI gate)

Known service caveats this design respects (2026-09-19 incident + briefing):
- ORDER BY on a raw Jev probability is unreliable; fuse with a lexical or
  structural rank instead of trusting one ordering (U3).
- batching many rows into one state changes judgments; keep states small
  and per-item (U7 chunks by ledger, never lane-batches).
- no evidence spans are returned; code locates the supporting text after
  the judgment (U2 `locate`).
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from jev_judge import DEFAULT_RECEIPT_DIR, healthcheck, run_ask  # noqa: E402

REPO_ROOT = HERE.parents[1]

# --------------------------------------------------------------------------
# U1 + U2: abstain policy and fast-path/fallback wrapper
# --------------------------------------------------------------------------


@dataclass
class AbstainPolicy:
    """Noul bands. Outside the band the answer stands; inside it, abstain."""

    low: float = 0.35
    high: float = 0.65

    def route(self, noul: float) -> str:
        if noul <= self.low:
            return "NO"
        if noul >= self.high:
            return "YES"
        return "ABSTAIN"


@dataclass
class Routing:
    decision: str  # YES | NO | ABSTAIN | FALLBACK_OUTAGE | FALLBACK_CHOICE
    noul: float | None = None
    answer: dict | None = None
    receipt_path: str | None = None
    note: str = ""


def judged(
    state: Any,
    instructions: str,
    *,
    label: str,
    policy: AbstainPolicy | None = None,
    min_choice_confidence: float = 0.60,
    receipt_dir: Path = DEFAULT_RECEIPT_DIR,
) -> Routing:
    """One noul judgment with outage fallback and abstain routing.

    FALLBACK_* routings mean: do exactly what the campaign did before Jev —
    escalate to the reasoning orchestrator (me) or the human root.
    """
    policy = policy or AbstainPolicy()
    health = healthcheck()
    if not health["healthy"]:
        return Routing(
            "FALLBACK_OUTAGE", note=f"healthcheck failed {health}",
        )
    receipt = run_ask(
        state,
        {"judgment": {"type": "noul", "instructions": instructions}},
        label=label,
        receipt_dir=receipt_dir,
    )
    answers = receipt["answer"].get("answers", receipt["answer"])
    noul = float(answers["judgment"]["noul"])
    route = policy.route(noul)
    return Routing(route, noul=noul, answer=answers, receipt_path=receipt.get("receipt_path"))


def judged_choice(
    state: Any,
    instructions: str,
    choices: list[str],
    *,
    label: str,
    min_confidence: float = 0.60,
    receipt_dir: Path = DEFAULT_RECEIPT_DIR,
) -> Routing:
    """One choice judgment; low concentration routes to fallback, not a coin flip."""
    health = healthcheck()
    if not health["healthy"]:
        return Routing("FALLBACK_OUTAGE", note=f"healthcheck failed {health}")
    receipt = run_ask(
        state,
        {"judgment": {"type": "choice", "instructions": instructions, "choices": choices}},
        label=label,
        receipt_dir=receipt_dir,
    )
    answers = receipt["answer"].get("answers", receipt["answer"])
    a = answers["judgment"]
    if float(a.get("confidence", 0.0)) < min_confidence:
        return Routing(
            "FALLBACK_CHOICE", answer=a, receipt_path=receipt.get("receipt_path"),
            note=f"choice confidence {a.get('confidence')} < {min_confidence}",
        )
    return Routing(a["choice"], answer=a, receipt_path=receipt.get("receipt_path"))


def locate(text: str, claim: str) -> str:
    """Code-side evidence span finder (Jev returns no spans; briefing U2).

    Crude but auditable: best overlapping token window around the claim's
    distinctive tokens. Returns the most claim-like line of text.
    """
    stop = set("the a an of to in for on with is are was were be this that it its".split())
    tokens = [t for t in re.findall(r"[a-zA-Z0-9_.%-]+", claim) if t.lower() not in stop]
    best, best_score = "", -1
    for line in text.splitlines():
        score = sum(1 for t in tokens if t.lower() in line.lower())
        if score > best_score:
            best, best_score = line.strip(), score
    return best or text[:120]


# --------------------------------------------------------------------------
# U3: rerank and fuse (reciprocal rank fusion; never trust one ordering)
# --------------------------------------------------------------------------


def reciprocal_rank_fusion(
    rankings: list[list[str]], k: int = 60,
) -> list[tuple[str, float]]:
    """Fuse several orderings; briefing: fusing beat Jev-alone and embedding-alone."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


def rerank_fuse(
    query: str,
    candidates: list[dict[str, str]],
    *,
    label: str,
    text_key: str = "text",
    id_key: str = "id",
    receipt_dir: Path = DEFAULT_RECEIPT_DIR,
) -> dict:
    """Lexical rank + Jev semantic rank -> RRF. Returns fused ranking + receipts.

    candidates: [{"id": path-or-key, "text": snippet}, ...] (<= 20 per call;
    batching beyond that changes judgments per the briefing).
    """
    if len(candidates) > 20:
        raise ValueError("keep states small: <= 20 candidates per call")
    # structural/lexical ranking (BM25-ish token overlap)
    q = re.findall(r"[a-z0-9]+", query.lower())
    lexical = sorted(
        candidates,
        key=lambda c: -sum(
            re.findall(r"[a-z0-9]+", c[text_key].lower()).count(t) for t in q
        ),
    )
    lexical_rank = [c[id_key] for c in lexical]
    health = healthcheck()
    jev_rank: list[str] | None = None
    receipts: list[str] = []
    if health["healthy"]:
        state = {"query": query, "candidates": {c[id_key]: c[text_key][:600] for c in candidates}}
        qs = {
            c[id_key]: {
                "type": "score",
                "instructions": (
                    f"Score this candidate's relevance to the query on an ordered "
                    f"scale: 0=irrelevant, 1=mentions the topic tangentially, "
                    f"2=substantively on-topic, 3=directly answers the query."
                ),
            }
            for c in candidates
        }
        receipt = run_ask(state, qs, label=label, receipt_dir=receipt_dir)
        receipts.append(receipt["receipt_path"])
        answers = receipt["answer"].get("answers", receipt["answer"])
        try:
            scored = sorted(
                ((float(answers[i]["score"]), i) for i in qs),
                reverse=True,
            )
            jev_rank = [i for _, i in scored]
        except (KeyError, TypeError, ValueError):
            jev_rank = None
    rankings = [lexical_rank] + ([jev_rank] if jev_rank else [])
    fused = reciprocal_rank_fusion(rankings)
    return {
        "fused": fused,
        "lexical_rank": lexical_rank,
        "jev_rank": jev_rank,
        "jev_used": jev_rank is not None,
        "receipts": receipts,
    }


# --------------------------------------------------------------------------
# U4: tiered eval judge (Braintrust-style accept / LLM / human)
# --------------------------------------------------------------------------


def eval_judge(
    artifact_text: str,
    claim_blocks: list[str],
    *,
    label: str,
    accept_at: float = 0.95,
    llm_band: float = 0.70,
    receipt_dir: Path = DEFAULT_RECEIPT_DIR,
) -> dict:
    """One request per claim block: needs_review + severity + failure_mode.

    Policy tiers (per briefing): noul >= accept_at accept; llm_band..accept_at
    -> reasoning-LLM judge; below -> human. Returns per-claim routing with
    the code-located evidence span (Jev returns none).
    """
    results = []
    for i, claim in enumerate(claim_blocks):
        state = {"artifact": artifact_text[:20000], "claim": claim}
        r = judged(
            state,
            "Considering only the artifact text, is this claim fully supported "
            "by the artifact without overstatement?",
            label=f"{label}__claim{i}",
            policy=AbstainPolicy(low=1 - accept_at, high=accept_at),
            receipt_dir=receipt_dir,
        )
        noul = r.noul
        if noul is None:
            tier = "FALLBACK"
        elif noul >= accept_at:
            tier = "ACCEPT"
        elif noul >= llm_band:
            tier = "LLM_JUDGE"
        else:
            tier = "HUMAN"
        results.append({
            "claim": claim,
            "support_probability": noul,
            "tier": tier,
            "evidence_span": locate(artifact_text, claim),
            "routing": r.decision,
            "receipt": r.receipt_path,
        })
    return {"label": label, "claims": results}


# --------------------------------------------------------------------------
# U5: harness reflex — safety gate for destructive/launch commands
# --------------------------------------------------------------------------

DESTRUCTIVE_MARKERS = (
    "rm -rf", "git reset --hard", "git push --force", "drop table", "shutdown",
    "modal app stop", "aws ec2 terminate", "docker system prune",
)


def safety_gate(command: str, *, label: str = "safety_gate") -> dict:
    """Exit-code-gated reflex (jgrep style) for CI / pre-launch hooks.

    Deterministic markers veto outright; Jev judges the grey zone; outage or
    abstain FAILS CLOSED (nonzero) — the campaign's own rule.
    """
    hit = [m for m in DESTRUCTIVE_MARKERS if m in command]
    if hit:
        return {"exit": 1, "reason": f"deterministic marker: {hit}", "tier": "VETO"}
    r = judged(
        {"command": command, "context": "benchmark campaign repo, receipts immutable"},
        "Could this command destroy evidence, launch paid compute, or mutate "
        "sealed receipts if run in the campaign repository?",
        label=label,
        policy=AbstainPolicy(low=0.30, high=0.70),
    )
    if r.decision == "YES":
        return {"exit": 1, "reason": "jev judged destructive", "tier": "VETO", "noul": r.noul}
    if r.decision == "NO":
        return {"exit": 0, "reason": "jev clear", "tier": "PASS", "noul": r.noul}
    return {
        "exit": 2, "reason": f"abstained/outage ({r.decision}); human review required",
        "tier": "REVIEW", "noul": r.noul,
    }


# --------------------------------------------------------------------------
# U6: action selection from structured state (per-lane, small state)
# --------------------------------------------------------------------------


def next_action(lane: str, facts: dict, *, choices: list[str]) -> Routing:
    return judged_choice(
        {"lane": lane, "facts": facts,
         "rule": "Pick the first uncleared blocker to a NEW native suite result."},
        f"Lane {lane}. Using only `facts`, pick the first uncleared blocker.",
        choices,
        label=f"next_action_{lane}",
    )


# --------------------------------------------------------------------------
# U7: bulk archive mining over the finish/ ledger corpus
# --------------------------------------------------------------------------


def mine_ledgers(
    root: Path,
    *,
    limit: int = 60,
    receipt_dir: Path = DEFAULT_RECEIPT_DIR,
) -> dict:
    """Per-ledger classification, one small state per file (never batched).

    Extracts {lane_hint, status_kind} per ledger JSON with receipts; falls
    back to keyword heuristics when the service is out.
    """
    ledgers = sorted(root.rglob("*.json"))
    out = []
    for path in ledgers[:limit]:
        try:
            text = path.read_text()[:4000]
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        lane_hint = next(
            (m.group(0).upper() for m in [re.search(r"\bE([1-9]|1[0-4])\b", path.name)] if m),
            "",
        )
        if not lane_hint:
            m = re.search(r"\bE([1-9]|1[0-4])\b", text[:500])
            lane_hint = ("E" + m.group(1)) if m else ""
        health = healthcheck()
        if health["healthy"]:
            r = judged_choice(
                {"file": path.name, "preview": text[:2000]},
                "Classify this campaign ledger's overall status.",
                ["terminal_complete", "in_progress", "blocked_external",
                 "blocked_paid_or_quota", "review_or_seal_record", "other"],
                label=f"mine_{path.name[:48]}",
                receipt_dir=receipt_dir,
            )
            status = r.decision if not r.decision.startswith("FALLBACK") else _keyword_status(text)
            used_jev = not r.decision.startswith("FALLBACK")
        else:
            status, used_jev = _keyword_status(text), False
        out.append({"path": str(path.relative_to(REPO_ROOT)), "lane": lane_hint,
                    "status": status, "jev_used": used_jev})
    return {"mined": len(out), "ledgers": out}


def _keyword_status(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ("blocked_external", "wait_provider", "private")):
        return "blocked_external"
    if any(k in t for k in ("quota", "vcpulimit", "pending", "unreserved")):
        return "blocked_paid_or_quota"
    if any(k in t for k in ("terminal", "complete", "scored")):
        return "terminal_complete"
    if any(k in t for k in ("sealed", "review", "validation")):
        return "review_or_seal_record"
    return "other"


if __name__ == "__main__":
    print(json.dumps(healthcheck(), indent=1))
    print(__doc__)
