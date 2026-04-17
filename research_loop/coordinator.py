"""Wave coordinator for the parallel-agent research loop.

Commands:
    coordinator.py status
        Current best, waves done, runs done, spend, learnings count.

    coordinator.py wave new --size N --phase 1|2|3
        Create a new wave directory under wave_briefs/, pick N hypotheses
        from queue.jsonl (or hypotheses_seed.md for wave 1), render one
        agent brief per hypothesis.

    coordinator.py wave ingest wave_NNN
        Read all *.result.json files in the wave directory, update
        best_recipe.yaml if any variant beat the current best by 10%+,
        append dead ends / confirmed findings to learnings.md, drain the
        queue, flag plateau if needed.

    coordinator.py queue add "<hypothesis one-liner>" --knobs group_size=32
        Add a new hypothesis to queue.jsonl.

    coordinator.py reset-for-scratch
        Emit a special agent brief with NO access to learnings.md, only
        the challenge + current best score + forbidden-knobs list.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
BEST = ROOT / "best_recipe.yaml"
LEARNINGS = ROOT / "learnings.md"
RESULTS = ROOT / "results.jsonl"
QUEUE = ROOT / "queue.jsonl"
SEED_HYPOTHESES = ROOT / "hypotheses_seed.md"
WAVE_DIR = ROOT / "wave_briefs"
VARIANT_DIR = ROOT / "variant_configs"
TEMPLATE = ROOT / "agent_brief_template.md"
SPEND_FILE = ROOT / "spend_usd.txt"

PROMOTION_DELTA = 0.10  # must beat current best by 10% to be promoted
PLATEAU_WAVE_COUNT = 3  # 3 consecutive waves with no new best → Phase 3 reset


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def dump_yaml(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False))


def load_best() -> dict:
    return load_yaml(BEST)


def save_best(data: dict) -> None:
    dump_yaml(BEST, data)


def load_queue() -> list[dict]:
    if not QUEUE.exists():
        return []
    return [json.loads(line) for line in QUEUE.read_text().splitlines() if line.strip()]


def save_queue(items: list[dict]) -> None:
    QUEUE.write_text("\n".join(json.dumps(i) for i in items) + ("\n" if items else ""))


def load_results() -> list[dict]:
    if not RESULTS.exists():
        return []
    return [json.loads(line) for line in RESULTS.read_text().splitlines() if line.strip()]


def load_seed_hypotheses() -> list[dict]:
    """Parse hypotheses_seed.md into structured list."""
    if not SEED_HYPOTHESES.exists():
        return []
    out: list[dict] = []
    pattern = re.compile(
        r"-\s+\*\*(H\d+)\*\*\s*\|\s*(\w+)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*)"
    )
    for line in SEED_HYPOTHESES.read_text().splitlines():
        m = pattern.match(line)
        if m:
            hid, cat, oneline, knobs, rationale = m.groups()
            out.append({
                "id": hid,
                "category": cat,
                "oneline": oneline.strip(),
                "knobs": knobs.strip(),
                "rationale": rationale.strip(),
            })
    return out


def current_spend() -> float:
    if not SPEND_FILE.exists():
        return 0.0
    try:
        return float(SPEND_FILE.read_text().strip())
    except ValueError:
        return 0.0


def list_waves() -> list[str]:
    if not WAVE_DIR.exists():
        return []
    return sorted(p.name for p in WAVE_DIR.iterdir() if p.is_dir() and p.name.startswith("wave_"))


def next_wave_name() -> str:
    waves = list_waves()
    if not waves:
        return "wave_001"
    last = int(waves[-1].split("_")[1])
    return f"wave_{last + 1:03d}"


# ── Commands ────────────────────────────────────────────────────────────

def cmd_status(_args) -> int:
    best = load_best()
    waves = list_waves()
    results = load_results()
    completed = [r for r in results if r.get("status") == "completed"]
    failed = [r for r in results if r.get("status") not in ("completed", None)]
    spend = current_spend()
    queue = load_queue()

    print("=" * 60)
    print(f"  Research Loop Status — {time.strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)
    print(f"  Current best:       {best.get('name')} @ {best.get('current_best_score')}")
    print(f"  Score metric:       {best.get('score_metric')}")
    print(f"  Seeds verified:     {best.get('n_seeds_verified', 0)}")
    print(f"  Waves completed:    {len(waves)}")
    print(f"  Runs (completed):   {len(completed)}")
    print(f"  Runs (failed):      {len(failed)}")
    print(f"  Queue length:       {len(queue)} (un-run hypotheses)")
    print(f"  Spend:              ${spend:.2f}")
    if os.environ.get("RESEARCH_LOOP_BUDGET_USD"):
        print(f"  Budget cap:         ${os.environ['RESEARCH_LOOP_BUDGET_USD']}")
    print()
    if completed:
        top = sorted(
            completed,
            key=lambda r: r.get("metrics", {}).get("last10_avg_accuracy", 0),
            reverse=True,
        )[:5]
        print("  Top 5 runs by last10_avg_accuracy:")
        for i, r in enumerate(top, 1):
            m = r.get("metrics", {})
            print(
                f"    {i}. {r['variant_id']:<20s}  "
                f"last10={m.get('last10_avg_accuracy', 0):.3f}  "
                f"peak={m.get('peak_accuracy', 0):.3f}  "
                f"(wave={r.get('wave', '?')})"
            )
    return 0


def cmd_wave_new(args) -> int:
    size = args.size
    phase = args.phase
    wave_name = next_wave_name()
    wave_path = WAVE_DIR / wave_name
    variant_path = VARIANT_DIR / wave_name
    wave_path.mkdir(parents=True, exist_ok=True)
    variant_path.mkdir(parents=True, exist_ok=True)

    best = load_best()
    queue = load_queue()

    # Seed from hypotheses_seed.md if both the queue and results.jsonl are empty
    # (first run of the loop, regardless of what wave number we ended up on).
    if not queue and not load_results():
        seeded = load_seed_hypotheses()
        print(f"Queue empty and no prior results — seeding {len(seeded)} hypotheses from hypotheses_seed.md")
        queue.extend(seeded)

    if not queue:
        print("ERROR: queue is empty. Add hypotheses via 'queue add' or re-seed.",
              file=sys.stderr)
        return 1

    picked = queue[:size]
    remaining = queue[size:]

    template = TEMPLATE.read_text()

    wave_readme_lines = [
        f"# Wave {wave_name.split('_')[1]} — Phase {phase}",
        "",
        f"Launched: {time.strftime('%Y-%m-%dT%H:%M:%S')}",
        f"Wave size: {len(picked)} agents",
        f"Current best: {best.get('name')} ({best.get('current_best_score')})",
        "",
        "## Hypotheses tested in this wave",
        "",
    ]

    for i, h in enumerate(picked, 1):
        variant_id = f"v{i:03d}"
        brief = (
            template
            .replace("{{variant_id}}", variant_id)
            .replace("{{wave_num}}", wave_name.split("_")[1])
            .replace("{{hypothesis_id}}", h.get("id", "H??"))
            .replace("{{hypothesis_oneline}}", h.get("oneline", ""))
            .replace("{{knobs_delta}}", h.get("knobs", ""))
            .replace("{{hypothesis_rationale}}", h.get("rationale", ""))
            .replace("{{current_best_score}}", str(best.get("current_best_score")))
            .replace("{{score_metric}}", best.get("score_metric", "last10_avg_accuracy"))
            .replace("{{parent_version}}", best.get("name", "baseline_v0"))
            .replace("{{max_runs}}", "1")
        )
        brief_path = wave_path / f"{variant_id}.md"
        brief_path.write_text(brief)
        wave_readme_lines.append(
            f"- **{variant_id}** ({h.get('id')}): {h.get('oneline')} → `{h.get('knobs')}`"
        )

    (wave_path / "README.md").write_text("\n".join(wave_readme_lines) + "\n")
    save_queue(remaining)

    print(f"Created {wave_name} with {len(picked)} agent briefs:")
    print(f"  {wave_path}")
    print(f"  Variant configs will land in: {variant_path}")
    print()
    print("Next: spawn one Claude Code agent per brief.")
    print("Each agent reads its brief + learnings.md, writes a variant YAML,")
    print("runs `python research_loop/run_one.py --config <variant>`,")
    print("then writes a .result.json in the same wave directory.")
    return 0


def cmd_wave_ingest(args) -> int:
    wave_name = args.wave_name
    wave_path = WAVE_DIR / wave_name
    if not wave_path.exists():
        print(f"ERROR: {wave_path} not found", file=sys.stderr)
        return 1

    results_files = sorted(wave_path.glob("*.result.json"))
    if not results_files:
        print(f"No result files in {wave_path}. Nothing to ingest.")
        return 1

    best = load_best()
    best_score = best.get("current_best_score") or 0.0
    metric_key = best.get("score_metric", "last10_avg_accuracy")

    ingested = 0
    new_best = None
    dead_ends: list[str] = []
    interesting: list[str] = []

    for rf in results_files:
        try:
            result = json.loads(rf.read_text())
        except json.JSONDecodeError:
            print(f"  skip {rf.name} (bad JSON)")
            continue
        ingested += 1
        status = result.get("status")
        if status != "completed":
            dead_ends.append(f"- {rf.stem}: {status} — {result.get('error', '')}")
            continue
        metrics = result.get("metrics", {})
        score = metrics.get(metric_key, 0.0)
        delta = score - best_score
        relative = (delta / best_score) if best_score else float("inf")

        if score > best_score * (1 + PROMOTION_DELTA):
            new_best = (rf, result, score)
        elif delta <= -0.05:
            dead_ends.append(
                f"- {rf.stem}: {metric_key}={score:.3f} vs {best_score:.3f} "
                f"(Δ={delta:+.3f}) — {result.get('notes', '')}"
            )
        else:
            interesting.append(
                f"- {rf.stem}: {metric_key}={score:.3f} (Δ={delta:+.3f})"
            )

    print(f"Ingested {ingested} result files from {wave_name}")

    if new_best:
        rf, result, score = new_best
        print(f"  NEW BEST: {rf.stem} → {score:.3f} (was {best_score:.3f})")
        cfg = result.get("config", {})
        best.update(cfg)
        best["name"] = f"{wave_name}_{rf.stem}"
        best["current_best_score"] = score
        best["parent"] = best.get("name")
        best["wave"] = wave_name
        save_best(best)
    else:
        print(f"  No new best (threshold: beat {best_score:.3f} by {PROMOTION_DELTA*100:.0f}%)")

    if dead_ends or interesting:
        with LEARNINGS.open("a") as f:
            f.write(f"\n\n## Wave {wave_name} ingest ({time.strftime('%Y-%m-%d')})\n")
            if dead_ends:
                f.write("\n### Dead ends\n")
                f.write("\n".join(dead_ends) + "\n")
            if interesting:
                f.write("\n### Neutral / marginal\n")
                f.write("\n".join(interesting) + "\n")

    return 0


def cmd_queue_add(args) -> int:
    queue = load_queue()
    item = {
        "id": f"H{len(queue) + 100:03d}",
        "category": args.category or "user",
        "oneline": args.oneline,
        "knobs": args.knobs or "",
        "rationale": args.rationale or "",
    }
    queue.append(item)
    save_queue(queue)
    print(f"Added {item['id']} to queue ({len(queue)} total)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status")

    wave = sub.add_parser("wave")
    wave_sub = wave.add_subparsers(dest="wave_cmd", required=True)
    wave_new = wave_sub.add_parser("new")
    wave_new.add_argument("--size", type=int, default=8)
    wave_new.add_argument("--phase", type=int, default=1, choices=[1, 2, 3])
    wave_ing = wave_sub.add_parser("ingest")
    wave_ing.add_argument("wave_name")

    q = sub.add_parser("queue")
    q_sub = q.add_subparsers(dest="q_cmd", required=True)
    q_add = q_sub.add_parser("add")
    q_add.add_argument("oneline")
    q_add.add_argument("--category", default=None)
    q_add.add_argument("--knobs", default=None)
    q_add.add_argument("--rationale", default=None)

    args = parser.parse_args()

    if args.cmd == "status":
        return cmd_status(args)
    if args.cmd == "wave":
        if args.wave_cmd == "new":
            return cmd_wave_new(args)
        if args.wave_cmd == "ingest":
            return cmd_wave_ingest(args)
    if args.cmd == "queue":
        if args.q_cmd == "add":
            return cmd_queue_add(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
