#!/usr/bin/env python3
"""
ZVF Program — Pillar-1 / M1 sweep orchestrator.

WHAT THIS DOES
  1. Reads sweep_config.yaml and expands it into a run MANIFEST (a list of
     run-cells, ~300-500 of them).
  2. For each cell, builds the EXACT shell command that launches the user's
     existing Tinker/Modal runner. It does NOT implement training. It only
     enumerates -> launches -> tracks status -> points at the per-step ZVF/GU
     each run already logs.
  3. Is RESUMABLE: cells whose result JSON already exists on disk (or whose tag
     already appears in master_results.csv) are skipped.

WHAT THIS DOES NOT DO
  * It never fabricates a metric, reward, ZVF, GU, or "win". Status is derived
    purely from process exit codes and the presence/parse-ability of the result
    JSON the real runner writes.
  * In --dry-run (the DEFAULT) it executes nothing: it prints the manifest and
    the literal commands it WOULD run.

USAGE
  python3 run_sweep.py                 # dry-run (default): print manifest + commands, run nothing
  python3 run_sweep.py --dry-run       # same as above, explicit
  python3 run_sweep.py --execute       # actually shell out to the real runners
  python3 run_sweep.py --execute --max-parallel 6
  python3 run_sweep.py --filter audit  # only cells whose tag contains 'audit'
  python3 run_sweep.py --limit 5       # cap number of cells (smoke test)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
# Repo root = .../tinker-rl-lab  (this file lives at zvf-program/sweep/)
REPO_ROOT = HERE.parents[1]
CONFIG_PATH = HERE / "sweep_config.yaml"


# ---------------------------------------------------------------------------
# Config + path helpers
# ---------------------------------------------------------------------------
def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve(p: str) -> Path:
    """Resolve a config path relative to the repo root unless already absolute."""
    pp = Path(p)
    return pp if pp.is_absolute() else (REPO_ROOT / pp)


def _apply_tier(fixed: dict, overrides: dict, tier: str) -> dict:
    merged = dict(fixed)
    merged.update(overrides.get(tier, {}) or {})
    return merged


# ---------------------------------------------------------------------------
# Manifest expansion
# ---------------------------------------------------------------------------
def _model_index(cfg: dict) -> dict:
    """short-name -> {family, short, model_id, tier}."""
    idx = {}
    for family, members in cfg["model_families"].items():
        for m in members:
            idx[m["short"]] = {**m, "family": family}
    return idx


def _enabled_frameworks(cfg: dict) -> list[str]:
    return [fw["name"] for fw in cfg["rl_frameworks"] if fw.get("enabled")]


def expand_manifest(cfg: dict) -> list[dict]:
    """Expand the declarative config into a flat list of run-cell dicts.

    Each cell carries everything needed to (a) build the launch command and
    (b) locate its result JSON. NO results are computed here.
    """
    midx = _model_index(cfg)
    frameworks = _enabled_frameworks(cfg)
    if not frameworks:
        raise SystemExit("No rl_frameworks are enabled in sweep_config.yaml.")
    seeds = cfg["seeds"]
    group_sizes = cfg["group_sizes"]
    difficulties = [d["name"] for d in cfg["difficulty_buckets"]]
    fixed = cfg["fixed"]
    tier_overrides = cfg.get("tier_overrides", {})
    design = cfg["sweep_design"]

    cells: list[dict] = []

    def make_cell(block, framework, m, seed, group_size, difficulty, hp, loss, tag):
        return {
            "tag": tag,
            "block": block,
            "framework": framework,
            "family": m["family"],
            "model_short": m["short"],
            "model_id": m["model_id"],
            "tier": m["tier"],
            "task": hp["task"],
            "loss": loss,
            "seed": seed,
            "group_size": group_size,
            "difficulty": difficulty,
            "lr": hp["lr"],
            "steps": hp["steps"],
            "rank": hp["rank"],
        }

    # ---- AUDIT block --------------------------------------------------------
    audit = design.get("audit", {})
    if audit.get("enabled"):
        full_tiers = set(audit["full_grid_tiers"])
        thin_tiers = set(audit["thin_grid_tiers"])
        base_g = audit["baseline_group_size"]
        base_d = audit["baseline_difficulty"]
        for framework in frameworks:
            fw_slug = framework.lower().replace("-", "")
            for short, m in midx.items():
                hp = _apply_tier(fixed, tier_overrides, m["tier"])
                loss = hp["loss"]
                if m["tier"] in full_tiers:
                    grid = [(s, g, d) for s in seeds for g in group_sizes for d in difficulties]
                elif m["tier"] in thin_tiers:
                    grid = [(s, base_g, base_d) for s in seeds]
                else:
                    continue
                for seed, g, d in grid:
                    tag = f"audit_{fw_slug}_{short}_G{g}_{d}_s{seed}"
                    cells.append(make_cell("audit", framework, m, seed, g, d, hp, loss, tag))

    # ---- MATCHED-COMPUTE block ---------------------------------------------
    mc = design.get("matched_compute", {})
    if mc.get("enabled"):
        # The arm names map to loss configs; matched_compute.py is authoritative
        # for rollout/token budgets. Here we just enumerate the cells so the
        # manifest is complete + resumable. We record arm in both loss + tag.
        for short in mc["models"]:
            if short not in midx:
                raise SystemExit(f"matched_compute model '{short}' not in model_families.")
            m = midx[short]
            hp = _apply_tier(fixed, tier_overrides, m["tier"])
            for arm in mc["arms"]:
                for seed in mc["seeds"]:
                    g = mc["group_size"]
                    d = mc["difficulty"]
                    tag = f"matched_{arm}_{short}_G{g}_{d}_s{seed}"
                    # Each matched arm runs on the single enabled framework.
                    cells.append(
                        make_cell("matched", frameworks[0], m, seed, g, d, hp, arm, tag)
                    )

    return cells


# ---------------------------------------------------------------------------
# Command building (the shell-out the user actually owns)
# ---------------------------------------------------------------------------
def build_command(cfg: dict, cell: dict) -> tuple[str, Path]:
    """Return (command_string, out_json_path) for a run-cell.

    Substitutes cell fields into runner.command_template. No execution here.
    """
    runner = cfg["runner"]
    kind = runner["kind"]
    rc = runner[kind]
    results_dir = resolve(rc["results_dir"])
    out_json = Path(
        rc["out_json_template"].format(results_dir=str(results_dir), tag=cell["tag"])
    )
    subst = {
        "interpreter": rc["interpreter"],
        "script": str(resolve(rc["script"])),
        "results_dir": str(results_dir),
        "out_json": str(out_json),
        # Audit-cell defaults for the arm-specific placeholders. These
        # come from the YAML `fixed` block; matched-compute cells
        # override them in matched_compute.py:arm_commands. We supply
        # them here so the audit enumeration never hits an unknown
        # placeholder error.
        "clip_low": cfg["fixed"]["clip_low"],
        "clip_high": cfg["fixed"]["clip_high"],
        "importance_level": cfg["fixed"]["importance_level"],
        "dynamic_sampling": cfg["fixed"]["dynamic_sampling"],
        **cell,
    }
    try:
        cmd = " ".join(rc["command_template"].split()).format(**subst)
    except KeyError as e:
        raise SystemExit(
            f"command_template references unknown placeholder {e} for cell {cell['tag']}"
        )
    return cmd, out_json


# ---------------------------------------------------------------------------
# Resume logic — what's already done
# ---------------------------------------------------------------------------
def load_completed_tags_from_master(cfg: dict) -> set[str]:
    """Tags already present (and completed) in the existing master_results.csv."""
    master = REPO_ROOT / "experiments" / "master_results.csv"
    done: set[str] = set()
    if not master.exists():
        return done
    try:
        with open(master, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("status") == "completed" and row.get("experiment_id"):
                    done.add(row["experiment_id"])
    except Exception:
        pass
    return done


def result_json_is_complete(path: Path, planned_steps: int | None = None) -> bool:
    """A cell counts as done iff its result JSON exists, parses, and is
    not a partial-run stub or a failure stub. We NEVER infer success
    from anything but the runner's own file.

    The previous version checked only "status == completed" OR
    "step_log is present", which silently treated a crashed-mid-write
    result JSON (file exists with a partial step_log) as done. Now we
    additionally require that the step_log length matches the planned
    number of steps when planned_steps is supplied, so a 5-of-30-step
    crash is not treated as completed.
    """
    if not path.exists():
        return False
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception:
        return False
    status = data.get("status")
    if status == "completed":
        # Authoritative: the runner itself reports completed.
        if planned_steps is not None and planned_steps > 0:
            step_log = data.get("step_log") or []
            if len(step_log) < planned_steps:
                # status="completed" but step_log is short. The
                # runner's status flag is not trustworthy; mark as
                # incomplete and let the resume loop re-run the cell.
                return False
        return True
    # Fallback: some runners omit status but write a step_log on success.
    if status is None and data.get("step_log"):
        step_log = data["step_log"]
        if planned_steps is not None and planned_steps > 0:
            if len(step_log) < planned_steps:
                # step_log present but truncated (crashed mid-run).
                return False
        return True
    return False


# ---------------------------------------------------------------------------
# Status ledger
# ---------------------------------------------------------------------------
LEDGER_COLS = ["tag", "block", "framework", "model_short", "status",
               "started_at", "finished_at", "returncode", "result_json", "command"]


def write_ledger(path: Path, rows: list[dict]) -> None:
    """Write the resume ledger atomically.

    The previous version opened the file in ``"w"`` mode (truncating
    on every call) and never fsync'd. A SIGKILL or a power loss
    between ``open`` and the last ``write`` would silently destroy
    the audit trail up to that point. We now:
      1. Write to a sibling temp file and atomically rename it over
         the real ledger.
      2. fsync the temp file before rename.
      3. fsync the directory after rename (so the rename itself is
         durable on POSIX).

    This makes the ledger a single source of truth that survives
    mid-write crashes; the resume logic can then read it
    confidently and never double-launch a cell.
    """
    import os as _os
    import tempfile as _tempfile
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = _tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with _os.fdopen(fd, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=LEDGER_COLS)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in LEDGER_COLS})
            f.flush()
            _os.fsync(f.fileno())
        _os.replace(tmp_path, path)
        # fsync the directory so the rename is durable
        dir_fd = _os.open(str(path.parent), _os.O_RDONLY)
        try:
            _os.fsync(dir_fd)
        finally:
            _os.close(dir_fd)
    except Exception:
        # Fall back to a direct write if the atomic dance fails
        # (e.g. the filesystem doesn't support rename). Still
        # better than silently losing the ledger.
        if _os.path.exists(tmp_path):
            _os.unlink(tmp_path)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=LEDGER_COLS)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in LEDGER_COLS})
            f.flush()
            _os.fsync(f.fileno())


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------
def run_one(cmd: str, out_json: Path, cell: dict) -> dict:
    started = datetime.now(timezone.utc).isoformat()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    # Guard against accidental shell metacharacters in cmd.
    # The substitution sources are the YAML (controlled) + the
    # audit cell fields (controlled), so this should never fire on
    # a clean config; the explicit check protects against a
    # fat-fingered YAML or a future field that lets untrusted input
    # through. A cell_runner.py that does not want shell features
    # can use the shlex=False mode by removing shell=True above.
    if not isinstance(cmd, str):
        raise TypeError(f"cmd must be str, got {type(cmd).__name__}")
    if any(c in cmd for c in ("\n", "\r", "\x00")):
        raise ValueError(
            f"cmd contains a forbidden newline/NUL byte (shell-injection "
            f"guard): {cmd[:80]!r}"
        )
    # Env scrubbing: pass a minimal env to the runner subprocess
    # rather than the full parent env. Without this, every secret
    # in the parent's env (API keys, AWS creds, ssh-agent, GH_TOKEN,
    # etc.) leaks into the runner's process. The minimal env keeps
    # PATH, HOME, LANG, LC_ALL, and TMPDIR (the runner needs them
    # to find python and the tinker client) and explicitly drops
    # everything else. Users who need to forward specific vars can
    # set SWEEP_FORWARD_ENV in their shell.
    import os as _os
    forward = set(
        (f.strip() for f in _os.environ.get("SWEEP_FORWARD_ENV", "").split(",") if f.strip())
    )
    _always_keep = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "USER", "SHELL")
    minimal_env = {k: v for k, v in _os.environ.items()
                    if k in _always_keep or k in forward}
    try:
        proc = subprocess.run(cmd, shell=True, cwd=str(REPO_ROOT),
                              timeout=4 * 3600, env=minimal_env)
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        return {**_ledger_base(cell, cmd, out_json), "status": "timeout",
                "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(),
                "returncode": "timeout:4h"}
    except Exception as e:  # pragma: no cover - defensive
        return {**_ledger_base(cell, cmd, out_json), "status": "error",
                "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(),
                "returncode": f"exc:{e}"}
    finished = datetime.now(timezone.utc).isoformat()
    # Status comes from the runner's own file, not from us guessing.
    ok = rc == 0 and result_json_is_complete(out_json)
    return {**_ledger_base(cell, cmd, out_json),
            "status": "completed" if ok else "failed",
            "started_at": started, "finished_at": finished, "returncode": rc}


def _ledger_base(cell: dict, cmd: str, out_json: Path) -> dict:
    return {"tag": cell["tag"], "block": cell["block"], "framework": cell["framework"],
            "model_short": cell["model_short"], "result_json": str(out_json), "command": cmd}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true",
                      help="(DEFAULT) print manifest + commands, execute nothing")
    mode.add_argument("--execute", action="store_true",
                      help="actually shell out to the real runners")
    ap.add_argument("--config", default=str(CONFIG_PATH))
    ap.add_argument("--filter", default="", help="substring filter on cell tag")
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help=(
            "cap the number of PENDING (not-yet-done) cells actually run. "
            "(0 = no cap). The dry-run header and the manifest still show the "
            "FULL enumeration; only the cells dispatched to subprocesses are "
            "capped. Use --filter to narrow the printed manifest, or "
            "--limit 5 --execute to dispatch at most 5 cells."
        ),
    )
    ap.add_argument("--max-parallel", type=int, default=4)
    ap.add_argument(
        "--no-resume",
        action="store_true",
        help=(
            "do not skip cells already completed on disk / in master CSV. "
            "WARNING: re-running completed cells OVERWRITES the prior result "
            "JSON. Use this only when you want to force a re-run; the default "
            "is to skip completed cells, which is almost always what you want."
        ),
    )
    ap.add_argument("--force-execute", action="store_true",
                    help="override the out-of-band guard: launch an enumeration "
                         "outside sweep_design.target_run_count. Use with care; "
                         "this is a deliberate 'I know what I'm doing' override.")
    args = ap.parse_args()

    execute = bool(args.execute)  # dry-run is the default whenever --execute is absent

    cfg = load_config(Path(args.config))
    cells = expand_manifest(cfg)
    if args.filter:
        cells = [c for c in cells if args.filter in c["tag"]]

    # --execute concurrency lock: a PID file under .run_sweep.lock
    # in the YAML's results dir (or cwd-relative) prevents two
    # parallel `run_sweep.py --execute` invocations from
    # double-launching the same cells and corrupting the resume
    # ledger. The previous version of this file had the lock
    # only on matched_compute.py; run_sweep.py was unprotected,
    # so two parallel sweeps from the same repo would silently
    # race on the ledger. The lock is acquired BEFORE the
    # band-check or any cell launch, and released via try/finally
    # so a SIGKILL during cell dispatch leaves a stale PID file
    # (detected via kill(pid, 0) on next start).
    sweep_lock: Path | None = None
    if execute and not getattr(args, "force_execute", False):
        lock_path = Path(args.config).resolve().parent / ".run_sweep.lock"
        sweep_lock = lock_path
        if lock_path.exists():
            try:
                old_pid = int(lock_path.read_text().strip())
                import os as _os
                try:
                    _os.kill(old_pid, 0)
                    print(
                        f"[error] Another run_sweep.py --execute is already "
                        f"running (PID {old_pid}, lock file {lock_path}). "
                        f"Refusing to launch. Pass --force-execute to override.",
                        file=sys.stderr,
                    )
                    return 1
                except ProcessLookupError:
                    print(
                        f"[warn] Stale lock file {lock_path} (PID {old_pid} no "
                        f"longer alive). Removing and continuing.",
                        file=sys.stderr,
                    )
                    lock_path.unlink()
            except (ValueError, OSError) as exc:
                print(
                    f"[warn] Could not read lock file {lock_path}: {exc}. "
                    f"Removing and continuing.",
                    file=sys.stderr,
                )
                lock_path.unlink()
        import os as _os
        lock_path.write_text(str(_os.getpid()))

    try:
        total_enumerated = len(cells)

        # ---- resume: figure out what's already done ----
        master_done = set() if args.no_resume else load_completed_tags_from_master(cfg)
        pending, skipped = [], []
        for c in cells:
            _, out_json = build_command(cfg, c)
            c["_out_json"] = out_json
            done = (not args.no_resume) and (
                c["tag"] in master_done
                or result_json_is_complete(out_json, planned_steps=c.get("steps"))
            )
            (skipped if done else pending).append(c)

        if args.limit and len(pending) > args.limit:
            pending = pending[: args.limit]

        # ---- band check (HARD guard when --execute is set) ----
        # When --execute is absent this is a dry-run, and the band check is
        # a stderr note (so a user inspecting a misconfigured sweep gets
        # clear feedback). When --execute is set, an out-of-band
        # enumeration is a hard error unless --force-execute is also
        # passed; otherwise a fat-fingered seed list silently launches
        # ~900 cells and the user discovers the cost only at the
        # accounting stage.
        band = cfg["sweep_design"]["target_run_count"]
        band_ok = band["min"] <= total_enumerated <= band["max"]
        band_note = "OK" if band_ok else f"OUTSIDE target band [{band['min']},{band['max']}]"
        if execute and not band_ok and not getattr(args, "force_execute", False):
            ap.error(
                f"total_enumerated={total_enumerated} is OUTSIDE the target band "
                f"[{band['min']}, {band['max']}]. Refusing to --execute with an "
                f"out-of-band enumeration. Pass --force-execute to override, or "
                f"edit sweep_design.target_run_count in the YAML."
            )

        # ---- runner script preflight (only when --execute is set) ----
        # Refuse to launch a 400-cell subprocess storm against a missing or
        # non-executable runner script. The user discovers the missing
        # script in seconds, not after the first 50 cells have failed.
        if execute:
            kind = cfg["runner"]["kind"]
            runner_script = Path(cfg["runner"][kind]["script"])
            if not runner_script.is_absolute():
                runner_script = (REPO_ROOT / runner_script).resolve()
            if not runner_script.exists():
                ap.error(
                    f"runner script does not exist: {runner_script}. The sweep "
                    f"harness only orchestrates; cell_runner.py is a user-owned "
                    f"shim (see sweep/README.md 'What you must provide')."
                )
            if not os.access(runner_script, os.X_OK) and not os.access(
                sys.executable, os.X_OK
            ):
                # The runner is dispatched via `python3 <script>`, so the
                # script's own exec bit is not strictly required; we only
                # need sys.executable (the interpreter) to be runnable.
                ap.error(
                    f"interpreter {sys.executable!r} is not executable; cannot "
                    f"launch cells."
                )

        # ---- summary header ----
        by_block: dict[str, int] = {}
        for c in cells:
            by_block[c["block"]] = by_block.get(c["block"], 0) + 1
        print("=" * 78)
        print(f"ZVF SWEEP  ({'EXECUTE' if execute else 'DRY-RUN — nothing will be launched'})")
        print("=" * 78)
        print(f"config            : {args.config}")
        print(f"runner.kind       : {cfg['runner']['kind']}")
        print(f"enabled frameworks: {_enabled_frameworks(cfg)}")
        print(f"total cells       : {total_enumerated}   [{band_note}]")
        for b, n in sorted(by_block.items()):
            print(f"   block {b:16s}: {n}")
        print(f"already completed : {len(skipped)} (skipped via resume)")
        print(f"pending to run    : {len(pending)}"
              + (f"  (capped to {args.limit})" if args.limit else ""))
        print("-" * 78)

        # ---- build the manifest rows (with commands) ----
        manifest_rows = []
        for c in cells:
            cmd, out_json = build_command(cfg, c)
            manifest_rows.append({
                "tag": c["tag"], "block": c["block"], "framework": c["framework"],
                "family": c["family"], "model_short": c["model_short"], "model_id": c["model_id"],
                "tier": c["tier"], "task": c["task"], "loss": c["loss"], "seed": c["seed"],
                "group_size": c["group_size"], "difficulty": c["difficulty"], "lr": c["lr"],
                "steps": c["steps"], "rank": c["rank"],
                "result_json": str(out_json), "command": cmd,
            })

        # ---- always write the manifest (even in dry-run) ----
        manifest_path = resolve(cfg["output"]["manifest_path"])
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w") as f:
            json.dump({
                "sweep_name": cfg["sweep_name"],
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "runner_kind": cfg["runner"]["kind"],
                "total_cells": total_enumerated,
                "by_block": by_block,
                "target_band": band, "band_ok": band_ok,
                "cells": manifest_rows,
            }, f, indent=2)
        print(f"manifest written  : {manifest_path}  ({total_enumerated} cells)")

        if not execute:
            print("-" * 78)
            print("COMMANDS THAT WOULD RUN (first 8 pending; full set in manifest):")
            for c in pending[:8]:
                cmd, _ = build_command(cfg, c)
                print(f"  $ {cmd}")
            if len(pending) > 8:
                print(f"  ... and {len(pending) - 8} more (see {manifest_path})")
            print("-" * 78)
            print("DRY-RUN complete. Nothing was launched. Re-run with --execute to launch.")
            return 0

        # ---- EXECUTE ----
        ledger_path = resolve(cfg["output"]["status_ledger"])
        ledger: list[dict] = []
        # seed ledger with skipped (already-done) cells for a complete record
        for c in skipped:
            cmd, out_json = build_command(cfg, c)
            ledger.append({**_ledger_base(c, cmd, out_json), "status": "completed",
                           "started_at": "", "finished_at": "", "returncode": 0})
        write_ledger(ledger_path, ledger)

        print(f"Launching {len(pending)} cells, max_parallel={args.max_parallel}")
        with ThreadPoolExecutor(max_workers=args.max_parallel) as pool:
            futs = {}
            for c in pending:
                cmd, out_json = build_command(cfg, c)
                futs[pool.submit(run_one, cmd, out_json, c)] = c
            for fut in as_completed(futs):
                row = fut.result()
                ledger.append(row)
                write_ledger(ledger_path, ledger)  # checkpoint after each cell
                print(f"  [{row['status']:9s}] {row['tag']} rc={row['returncode']}")

        n_ok = sum(1 for r in ledger if r["status"] == "completed")
        print("-" * 78)
        print(f"DONE. {n_ok}/{len(ledger)} cells completed. Ledger: {ledger_path}")
        print("Next: python3 aggregate_sweep.py   (joins ZVF/GU with held-out scores)")
        return 0

    finally:
        if sweep_lock is not None:
            sweep_lock.unlink()


if __name__ == "__main__":
    sys.exit(main())
