#!/usr/bin/env python3
"""
MiniMax-M3 long-horizon autoresearch orchestrator.

Drives MiniMax M3 (via the Claude Agent SDK against MiniMax's Anthropic-compatible
endpoint) through the deli-autoresearch protocol: fresh session per iteration,
state injected via files, stall detection + forced pivots, wall-clock budget,
and a hard programmatic guardrail (can_use_tool) that keeps the worker inside an
isolated git worktree and blocks destructive / outward-facing operations.

Model + auth come from the environment (see .env.minimax): ANTHROPIC_BASE_URL,
ANTHROPIC_AUTH_TOKEN, ANTHROPIC_MODEL=MiniMax-M3[1m]. This process only reads them
indirectly — it never prints or persists the token.

Usage (see run.sh):
  python orchestrator.py --worktree /path/to/worktree --state-dir ./state \
      --max-hours 35 --pillars all --round-cap 40
"""
import argparse
import asyncio
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from claude_agent_sdk import (
    ClaudeSDKClient,
    ClaudeAgentOptions,
    PermissionResultAllow,
    PermissionResultDeny,
    ResultMessage,
)

# --------------------------------------------------------------------------- #
# Time / logging helpers
# --------------------------------------------------------------------------- #
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def mono() -> float:
    return time.monotonic()


async def single_user_message(text: str):
    """Streaming input: can_use_tool requires the prompt as an AsyncIterable."""
    yield {"type": "user", "message": {"role": "user", "content": text}}


class State:
    def __init__(self, state_dir: Path, worktree: Path):
        self.dir = state_dir
        self.worktree = worktree
        self.logs = state_dir.parent / "logs"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.logs.mkdir(parents=True, exist_ok=True)
        self.progress_path = self.dir / "progress.json"
        self.iter_log = self.dir / "iteration_log.jsonl"
        self.dir_tried = self.dir / "directions_tried.json"
        self.hb = self.dir / "heartbeat.json"
        self.work_log = self.logs / "orchestrator.jsonl"

    def log(self, level: str, event: str, **detail):
        line = {"ts": now_iso(), "source": "orchestrator", "level": level,
                "event": event, **detail}
        with self.work_log.open("a") as f:
            f.write(json.dumps(line) + "\n")
        print(f"[{level:8}] {event}: {json.dumps(detail)[:300]}", flush=True)

    def heartbeat(self, **extra):
        self.hb.write_text(json.dumps({"ts": now_iso(), "mono": mono(), **extra}))

    def load_progress(self) -> dict:
        if self.progress_path.exists():
            return json.loads(self.progress_path.read_text())
        return {"iteration": 0, "status": "init", "stale_count": 0,
                "total_cost_usd": 0.0, "pillar_idx": 0, "findings": 0}

    def save_progress(self, p: dict):
        self.progress_path.write_text(json.dumps(p, indent=2))

    def load_tried(self) -> list:
        if self.dir_tried.exists():
            return json.loads(self.dir_tried.read_text())
        return []

    def save_tried(self, tried: list):
        self.dir_tried.write_text(json.dumps(tried, indent=2))

    def append_iter(self, rec: dict):
        with self.iter_log.open("a") as f:
            f.write(json.dumps(rec) + "\n")


# --------------------------------------------------------------------------- #
# Guardrail: keep the worker inside the worktree, block dangerous / outward ops
# --------------------------------------------------------------------------- #
BASH_DENY = [
    r"\bgit\s+push\b", r"\bgit\s+remote\b", r"push\s+--?f", r"--force\b",
    r"\bgit\s+reset\s+--hard\b", r"\bgit\s+clean\s+-[a-z]*f", r"filter-branch",
    r"\bgh\s", r"\brm\s+-[rf]{1,2}\b", r"\bdd\s+if=", r"\bmkfs", r":\(\)\s*\{",
    r"\bsudo\b", r"\bshutdown\b", r"\breboot\b", r">\s*/dev/sd",
    r"MINIMAX_API_KEY", r"ANTHROPIC_AUTH_TOKEN", r"\.env\.minimax",
    r"\bprintenv\b", r"\benv\s*\|", r"\bset\s*\|",
    r"wandb\s+(login|sync)", r"huggingface-cli\s+(login|upload)", r"\bhf\s+auth",
    r"\b(curl|wget)\b.*(-d|--data|-F|--form|-T|--upload-file)",
]
BASH_DENY_RE = [re.compile(p, re.IGNORECASE) for p in BASH_DENY]

# Tools that are always safe to auto-allow (search/plan + any MCP tool).
SAFE_PREFIX = ("mcp__",)
SAFE_TOOLS = {
    "Glob", "Grep", "LS", "TodoWrite", "BashOutput",
    "WebSearch", "WebFetch", "Task", "ExitPlanMode", "KillShell",
}
# Path-bearing tools whose paths we coerce into the worktree (MiniMax M3 tends to
# emit hallucinated absolute paths; we rewrite them rather than fail the call).
PATH_TOOLS = {"Write", "Edit", "MultiEdit", "NotebookEdit", "Read", "NotebookRead"}
REPO_DIRS = ("experiments", "paper", "scripts", "figures", "reports", "atropos",
             "skyrl", "openrlhf", "verl", "trl_integrations", "unified", "tinkerrl",
             "docs", "tests", "reproducibility", "submission")

_WORKTREE = None   # set in main()
_DENY_LOG = None   # set in main()


def _log_deny(tool: str, reason: str, info):
    if _DENY_LOG:
        try:
            with open(_DENY_LOG, "a") as f:
                f.write(json.dumps({"ts": now_iso(), "tool": tool,
                                    "reason": reason[:200], "info": str(info)[:200]}) + "\n")
        except Exception:
            pass


def coerce_into_worktree(fp: str):
    """Return (path, changed). Foreign/absolute paths are remapped into the worktree,
    preserving a repo-relative suffix when we can recognize one, else by basename."""
    if not fp:
        return fp, False
    if not os.path.isabs(fp):
        return fp, False  # relative → resolves against cwd (the worktree)
    rp = os.path.realpath(fp)
    if rp == _WORKTREE or rp.startswith(_WORKTREE + os.sep):
        return fp, False
    parts = fp.replace("\\", "/").split("/")
    for i, seg in enumerate(parts):
        if seg in REPO_DIRS:
            return os.path.join(_WORKTREE, "/".join(parts[i:])), True
    return os.path.join(_WORKTREE, os.path.basename(fp)), True


async def guardrail(tool_name, input_data, context):
    # MCP tools (papers/docs/search) + known safe search/plan tools: allow untouched.
    if tool_name.startswith(SAFE_PREFIX) or tool_name in SAFE_TOOLS:
        return PermissionResultAllow(updated_input=input_data)

    # Bash: block dangerous / outward-facing operations.
    if tool_name == "Bash":
        cmd = input_data.get("command", "")
        for rx in BASH_DENY_RE:
            if rx.search(cmd):
                _log_deny(tool_name, f"pattern:{rx.pattern}", cmd)
                return PermissionResultDeny(
                    message=(f"Blocked by guardrail (pattern {rx.pattern!r}). "
                             "Not allowed: push/remote/force, history rewrite, gh, "
                             "rm -rf, secret exposure, external uploads, "
                             "wandb/hf login+upload. Work locally in the worktree."),
                    interrupt=False)
        return PermissionResultAllow(updated_input=input_data)

    # Path-bearing tools: protect secrets, keep everything inside the worktree.
    if tool_name in PATH_TOOLS:
        key = "file_path" if "file_path" in input_data else \
              ("path" if "path" in input_data else None)
        fp = (input_data.get(key) if key else "") or ""
        base = os.path.basename(fp)
        if base.startswith(".env") or base.endswith(".pem") or "/.claude" in fp:
            _log_deny(tool_name, "secret_access", fp)
            return PermissionResultDeny(message="Refusing to touch secrets/creds.")
        newfp, changed = coerce_into_worktree(fp)
        if changed and key:
            _log_deny(tool_name, "path_coerced", f"{fp} -> {newfp}")
            ni = dict(input_data)
            ni[key] = newfp
            return PermissionResultAllow(updated_input=ni)
        return PermissionResultAllow(updated_input=input_data)

    # Unknown/other SDK built-ins: allow.
    return PermissionResultAllow(updated_input=input_data)


# --------------------------------------------------------------------------- #
# MCP servers: reuse the user's already-configured LOCAL servers (with their
# API keys) straight from ~/.claude.json — nothing is copied or printed.
# --------------------------------------------------------------------------- #
def load_paper_mcp_servers() -> dict:
    servers = {}
    cfg_path = Path.home() / ".claude.json"
    if not cfg_path.exists():
        return servers
    try:
        cfg = json.loads(cfg_path.read_text())
    except Exception:
        return servers
    wanted = {"firecrawl", "brave-search", "serper"}  # local, paper/web search
    for name, s in (cfg.get("mcpServers") or {}).items():
        if name in wanted and s.get("command"):
            servers[name] = {
                "type": "stdio",
                "command": s["command"],
                "args": s.get("args", []),
                "env": s.get("env", {}) or {},
            }
    return servers


# --------------------------------------------------------------------------- #
# Progress measurement (objective, done by orchestrator not the worker)
# --------------------------------------------------------------------------- #
def worktree_changed_files(worktree: Path) -> int:
    try:
        out = subprocess.run(["git", "status", "--porcelain"], cwd=worktree,
                             capture_output=True, text=True, timeout=60).stdout
        return len([l for l in out.splitlines() if l.strip()])
    except Exception:
        return -1


def findings_count(worktree: Path) -> int:
    fp = worktree / "AUTORESEARCH_FINDINGS.jsonl"
    if fp.exists():
        return sum(1 for _ in fp.open())
    return 0


# --------------------------------------------------------------------------- #
# Prompt construction
# --------------------------------------------------------------------------- #
def build_prompt(task_spec: str, pillar: dict, tried: list, last_summary: str,
                 pivot: bool, iteration: int, worktree: Path,
                 round_cap_hint: int = 40) -> str:
    tried_txt = "\n".join(f"  - {t}" for t in tried[-25:]) or "  (none yet)"
    pivot_note = ""
    if pivot:
        pivot_note = (
            "\n[FORCED PIVOT] The last iterations stalled. Change a STRUCTURAL "
            "constraint, not tactical parameters: switch to a different pillar, a "
            "different data slice, a different analysis method, or a different "
            "paper section. Your new direction MUST differ from every tried one.\n")
    return f"""You are an autonomous research engineer improving **TinkerRL-Bench**
(a NeurIPS-targeted RL-for-LLMs benchmark) toward award-caliber quality. This is
iteration {iteration}. You run UNATTENDED — never ask the user anything, never end
on a question; resolve ambiguity yourself and just do the work.

## Your working directory
{worktree}
This is an isolated git worktree. All existing repo data is here (experiments/results/*.tsv,
*.json reward traces, paper/, ELEVATION_ROADMAP.md). WRITE ONLY inside this directory.
Do NOT git push, use gh, touch secrets, or run destructive commands (a guardrail blocks these).

## Frontier-model reasoning (consult EACH iteration)
If `./FRONTIER_INSIGHTS.md` exists in this worktree, READ it first — it holds the latest
ChatGPT-Pro-Extended + Gemini-Deep-Think reasoning on these 4 pillars, refreshed periodically from
a live 30-call cross-examination. Fold in any relevant argument, invention, or falsification it
raises (and attribute as "(frontier synthesis)"); use it to sharpen this iteration's deliverable.

## Real experiments (TINKER_API_KEY is set — use as needed)
You MAY run REAL, SMALL Tinker training/eval to generate NEW measured data when a claim needs it:
use the `tinker` SDK (models incl. meta-llama/Llama-3.2-3B, Qwen3, Nemotron, Kimi) or the `grpo_*.py`
drivers in this worktree. Keep runs CHEAP — small model, few steps, 1-3 seeds — log approximate cost,
and background long runs (poll, don't block). Analyse existing results first; only launch a Tinker run
when it materially strengthens a pillar's evidence.

## Current focus — Pillar {pillar['id']}: {pillar['title']}
{pillar['detail']}

Deliverables to build/extend this iteration (concrete artifacts, not a plan):
{pillar['deliverables']}

## Directions already tried (do NOT repeat)
{tried_txt}
{pivot_note}
## Last iteration summary
{last_summary or '(first iteration)'}

## How to get papers / citations
Use MCP tools as needed. Available: firecrawl research tools
(mcp__firecrawl__firecrawl_research_search_papers, ..._read_paper, ..._related_papers)
for arXiv/paper retrieval; brave-search / serper for web search; context7 for library docs.
Fall back to WebSearch/WebFetch if an MCP call fails. VERIFY every citation (title, authors,
year, venue) before using it — zero fabricated references. Prefer 2025–2026 GRPO/RL papers.

## Rules (deli protocol)
- ACT FAST: within your first ~3 turns, create the analysis script IN THE WORKTREE and run
  it on the real data. Write files incrementally and append a finding the moment you have
  one. Do NOT spend the whole turn budget reading/searching — bias hard toward producing
  artifacts early, because the session is capped at {round_cap_hint} turns.
- Produce VERIFIABLE artifacts this session: run the analysis, save real outputs, compile
  LaTeX if you touch paper/. Validate (execute scripts / latexmk) before finishing.
- Keep each written file <= 300 lines; touch at most ~5 large files this iteration.
- Append one JSON line per concrete finding to ./AUTORESEARCH_FINDINGS.jsonl in this
  worktree: {{"ts","pillar","claim","evidence_path","citation_ok"}}.
- End when the deliverable is meaningfully advanced (not when "ready to start").

Begin now. Make measurable progress on Pillar {pillar['id']} and record findings."""


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
async def run_iteration(prompt: str, worktree: Path, mcp_servers: dict,
                        round_cap: int, st: State) -> dict:
    options = ClaudeAgentOptions(
        cwd=str(worktree),
        permission_mode="default",          # unapproved tools fall to guardrail
        can_use_tool=guardrail,
        max_turns=round_cap,
        setting_sources=["project", "local"],  # project .mcp.json (context7) + enable flag
        mcp_servers=mcp_servers,               # + firecrawl/brave/serper (with keys)
        system_prompt={"type": "preset", "preset": "claude_code",
                       "append": (
                           "You are running UNATTENDED in a long-horizon research loop. "
                           f"Your working directory IS {worktree} (a git worktree). ALWAYS "
                           "use paths RELATIVE to it (e.g. experiments/results/x.tsv, "
                           "scripts/y.py) — NEVER absolute paths. Be decisive, produce real "
                           "artifacts every session, never ask questions, never wait for "
                           "confirmation.")},
    )
    result = {"ok": False, "cost": 0.0, "turns": 0, "subtype": "", "summary": "", "usage": {}}
    try:
        # ClaudeSDKClient keeps the permission channel open so can_use_tool works
        # across the whole session (one-shot query() closes it → "Stream closed").
        async with ClaudeSDKClient(options=options) as client:
            await client.query(prompt)
            async for message in client.receive_response():
                if isinstance(message, ResultMessage):
                    result["ok"] = not message.is_error
                    result["cost"] = message.total_cost_usd or 0.0
                    result["turns"] = message.num_turns or 0
                    result["subtype"] = message.subtype or ""
                    result["usage"] = message.usage or {}
                    result["summary"] = (message.result or "")[:1200] if not message.is_error \
                        else f"ERROR {message.subtype}: {message.errors}"
    except Exception as e:  # runtime fragility: never let one iteration kill the loop
        msg = str(e)
        if "maximum number of turns" in msg:
            # Expected soft stop for a bounded work session (deli round cap) —
            # artifacts written before the cap still count as progress.
            result["subtype"] = result["subtype"] or "error_max_turns"
            result["summary"] = result["summary"] or "hit round cap (partial work saved)"
        else:
            result["summary"] = f"EXCEPTION: {type(e).__name__}: {e}"
            st.log("error", "iteration_exception", err=msg[:300])
    return result


async def main():
    global _WORKTREE
    ap = argparse.ArgumentParser()
    ap.add_argument("--worktree", required=True)
    ap.add_argument("--state-dir", default="./state")
    ap.add_argument("--task-spec", default="./state/task_spec.md")
    ap.add_argument("--max-hours", type=float, default=35.0)
    ap.add_argument("--max-iterations", type=int, default=10_000)
    ap.add_argument("--max-cost-usd", type=float, default=0.0)  # 0 = no cap
    ap.add_argument("--round-cap", type=int, default=40)
    ap.add_argument("--pillars", default="all")  # "all" or e.g. "1,2"
    ap.add_argument("--pillars-file", default="")  # JSON list overriding load_pillars()
    args = ap.parse_args()

    global _DENY_LOG
    worktree = Path(os.path.realpath(args.worktree))
    _WORKTREE = str(worktree)
    st = State(Path(args.state_dir).resolve(), worktree)
    _DENY_LOG = str(st.logs / "guardrail_denials.jsonl")

    # Pillars (rich detail lives in task_spec.md; keep a compact registry here)
    task_spec = Path(args.task_spec).read_text() if Path(args.task_spec).exists() else ""
    if args.pillars_file and Path(args.pillars_file).exists():
        PILLARS = json.loads(Path(args.pillars_file).read_text())
    else:
        PILLARS = load_pillars(task_spec)
    if args.pillars != "all":
        keep = {int(x) for x in args.pillars.split(",") if x.strip()}
        PILLARS = [p for p in PILLARS if p["id"] in keep]
    if not PILLARS:
        st.log("error", "no_pillars", detail="nothing to run"); return

    mcp_servers = load_paper_mcp_servers()
    st.log("info", "startup", worktree=str(worktree),
           pillars=[p["id"] for p in PILLARS],
           mcp=list(mcp_servers.keys()) + ["context7(project)"],
           max_hours=args.max_hours)

    prog = st.load_progress()
    tried = st.load_tried()
    prog["status"] = "running"
    st.save_progress(prog)

    start = mono()
    deadline = start + args.max_hours * 3600
    stop_file = st.dir / "STOP"

    while True:
        st.heartbeat(iteration=prog["iteration"], stale=prog["stale_count"])
        # ---- termination checks
        if stop_file.exists():
            st.log("info", "stop_requested", detail="STOP file present"); break
        if mono() >= deadline:
            st.log("info", "budget_reached", detail="max wall-clock hit"); break
        if prog["iteration"] >= args.max_iterations:
            st.log("info", "iteration_cap"); break
        if args.max_cost_usd and prog["total_cost_usd"] >= args.max_cost_usd:
            st.log("info", "cost_cap", spent=prog["total_cost_usd"]); break

        # ---- pick pillar: round-robin for balanced coverage; stall sets a pivot flag
        # that instructs a structural change (handled in the prompt), not just a jump.
        pivot = prog["stale_count"] >= 2
        prog["pillar_idx"] = prog["iteration"] % len(PILLARS)
        pillar = PILLARS[prog["pillar_idx"]]

        files_before = worktree_changed_files(worktree)
        find_before = findings_count(worktree)

        prompt = build_prompt(task_spec, pillar, tried,
                              prog.get("last_summary", ""), pivot,
                              prog["iteration"] + 1, worktree, args.round_cap)
        st.log("info", "iteration_start", n=prog["iteration"] + 1,
               pillar=pillar["id"], pivot=pivot,
               elapsed_h=round((mono() - start) / 3600, 2))

        res = await run_iteration(prompt, worktree, mcp_servers, args.round_cap, st)

        files_after = worktree_changed_files(worktree)
        find_after = findings_count(worktree)
        new_files = max(0, files_after - files_before) if files_after >= 0 else 0
        new_finds = max(0, find_after - find_before)
        # Progress = real artifacts produced, even if the round cap was hit.
        progressed = (new_files > 0 or new_finds > 0)

        prog["iteration"] += 1
        prog["total_cost_usd"] = round(prog["total_cost_usd"] + res["cost"], 4)
        prog["last_summary"] = res["summary"]
        prog["findings"] = find_after
        if progressed:
            prog["stale_count"] = 0
            tried.append(f"iter{prog['iteration']} P{pillar['id']} "
                         f"{pillar['key']}: {res['summary'][:120]}")
        else:
            prog["stale_count"] += 1
        if prog["stale_count"] >= 4:
            st.log("warn", "structurally_stuck",
                   detail="4+ stalls; flag for human attention")

        st.heartbeat(iteration=prog["iteration"], stale=prog["stale_count"])
        st.save_progress(prog)
        st.save_tried(tried)
        st.append_iter({
            "ts": now_iso(), "iteration": prog["iteration"], "pillar": pillar["id"],
            "ok": res["ok"], "subtype": res["subtype"], "turns": res["turns"],
            "cost_usd_est": res["cost"], "tokens": res.get("usage", {}),
            "new_files": new_files, "new_findings": new_finds,
            "progressed": progressed, "stale_count": prog["stale_count"],
            "summary": res["summary"][:500],
        })
        st.log("info", "iteration_done", n=prog["iteration"], ok=res["ok"],
               progressed=progressed, new_files=new_files, new_findings=new_finds,
               cost_usd=res["cost"], total_cost=prog["total_cost_usd"],
               stale=prog["stale_count"])

    prog["status"] = "stopped"
    st.save_progress(prog)
    st.log("info", "shutdown", iterations=prog["iteration"],
           total_cost=prog["total_cost_usd"],
           elapsed_h=round((mono() - start) / 3600, 2))


def load_pillars(task_spec: str) -> list:
    """Compact pillar registry; 'detail' is a short brief, full context is in task_spec.md
    which the worker also reads from its worktree (ELEVATION_ROADMAP.md)."""
    return [
        {"id": 1, "key": "scaling_laws", "title": "GRPO Scaling Laws",
         "detail": "Fit the saturation model R(t)=R_max*(1-e^{-lambda t}) to reward "
                   "traces across 4B-671B (Qwen3-8B, Qwen3.5-4B, Llama-3.1-8B, "
                   "DeepSeek-V3.1, Nemotron-120B). Derive R_max, lambda, t_80=-ln(0.2)/lambda. "
                   "Test the three-phase hypothesis (arXiv 2507.18014); document where "
                   "Nemotron-120B's collapse violates it. Data: experiments/results/*.json, "
                   "modal_results_all.json.",
         "deliverables": "scripts/scaling_law_fit.py; experiments/results/scaling_law_fits.tsv; "
                         "paper/sections/scaling_laws.tex; figures/scaling_law_fit.pdf"},
        {"id": 2, "key": "zvf", "title": "Zero-Variance Fraction Diagnostic",
         "detail": "Elevate ZVF (fraction of groups with identical rewards) to a first-class "
                   "cross-library diagnostic vs AERO (arXiv 2602.14338). Report ZVF across all "
                   "experiments; correlate with training failure (Nemotron-120B collapse, "
                   "tool-use 0%). Data: experiments/results/groupsize_zvf_sweep.{tsv,json}, "
                   "zvf_partial_correlations.tsv, tinker_gsm8k_zvf_*.json.",
         "deliverables": "scripts/zvf_diagnostic.py; experiments/results/zvf_summary.tsv; "
                         "paper/sections/zvf.tex; figures/zvf_vs_failure.pdf"},
        {"id": 3, "key": "group_size", "title": "Group Size G=4 vs G=32",
         "detail": "Analyze G=4 vs G=32 in relation to 'It Takes Two: Your GRPO Is Secretly "
                   "DPO' (arXiv 2510.00977, G=2~=G=16 claim). Plot reward vs group size; test "
                   "whether G=4~=G=32 at broader scale. Data: experiments/results/"
                   "group_size_token_normalized.tsv, groupsize_zvf_sweep.tsv.",
         "deliverables": "scripts/group_size_analysis.py; experiments/results/group_size_effect.tsv; "
                         "paper/sections/group_size.tex; figures/group_size.pdf"},
        {"id": 4, "key": "length_bias", "title": "Length Bias (Dr. GRPO / MAD GRPO)",
         "detail": "Measure response-length bias: per-step mean length vs reward; Spearman "
                   "correlation; Dr. GRPO signature = increasing length + plateau/decreasing "
                   "reward. Data: experiments/results/drgrpo_*.json, run logs / CSVs.",
         "deliverables": "scripts/length_bias.py; experiments/results/length_bias.tsv; "
                         "paper/sections/length_bias.tex; figures/length_vs_reward.pdf"},
    ]


if __name__ == "__main__":
    asyncio.run(main())
