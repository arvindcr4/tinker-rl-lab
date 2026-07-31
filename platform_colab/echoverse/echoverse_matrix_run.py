#!/usr/bin/env python3
"""Run Echoverse sweeps from Colab/CLI with full task coverage and multi-seed replication.

Goals aligned to reviewer concerns:

* Run complete env × task grids (or bounded smoke subsets by request)
* Use multiple random seeds consistently
* Optionally upload reproducible summaries to W&B
* Pull missing grounding DB assets from HF and verify each task trajectory
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Tuple


LOG = logging.getLogger("echoverse-matrix")

EMBEDDED_ENVS = [
    "echostay",
    "echoforge",
    "datepickers",
    "datepickers_ood",
    "nested_filter",
    "nested_filter_ood",
]


@dataclass
class TaskResult:
    env: str
    seed: int
    task_id: str
    score: Optional[float]
    pass_fail: Optional[str]
    reason: str
    duration_s: Optional[float]
    verify_rc: int
    final_db: Optional[str] = None
    output: Optional[str] = None

    def as_json(self) -> Dict[str, Any]:
        return {
            "env": self.env,
            "seed": self.seed,
            "task_id": self.task_id,
            "score": self.score,
            "pass_fail": self.pass_fail,
            "reason": self.reason,
            "duration_s": self.duration_s,
            "verify_rc": self.verify_rc,
            "final_db": self.final_db,
            "output": self.output,
        }


@dataclass
class RunResult:
    env: str
    seed: int
    run_id: str
    task_count: int
    tasks_requested: int
    batch_rc: int
    batch_runtime_s: float
    batch_output: str
    verified: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    results: List[TaskResult] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return (self.passed / self.verified) if self.verified else 0.0

    def as_json(self) -> Dict[str, Any]:
        return {
            "env": self.env,
            "seed": self.seed,
            "run_id": self.run_id,
            "task_count": self.task_count,
            "tasks_requested": self.tasks_requested,
            "batch_rc": self.batch_rc,
            "batch_runtime_s": self.batch_runtime_s,
            "batch_output": self.batch_output,
            "verified": self.verified,
            "passed": self.passed,
            "failed": self.failed,
            "errors": self.errors,
            "pass_rate": self.pass_rate,
            "results": [r.as_json() for r in self.results],
        }


def _parse_csv_ints(value: str) -> List[int]:
    if not value:
        return []
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def _shorten(text: str, limit: int = 4000) -> str:
    if not text:
        return ""
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _run_cmd(cmd: List[str], cwd: Optional[Path] = None, env: Optional[Dict[str, str]] = None) -> Tuple[int, str]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        capture_output=True,
        text=True,
    )
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    return proc.returncode, out


def ensure_repo(repo_dir: Path, repo_url: str, branch: Optional[str] = None) -> None:
    """Clone or update the Echoverse repo.

    If the checkout becomes inconsistent, fall back to a clean clone.
    """
    if repo_dir.exists():
        LOG.info("Echoverse repo exists at %s; updating.", repo_dir)
        fetch_cmd = ["git", "-C", str(repo_dir), "fetch", "origin", "--prune"]
        rc, out = _run_cmd(fetch_cmd)
        if rc != 0:
            LOG.warning("git fetch failed in %s:\n%s", repo_dir, out.strip())
            LOG.info("Falling back to fresh clone.")
            shutil.rmtree(repo_dir)
        elif branch:
            rc, out = _run_cmd(
                ["git", "-C", str(repo_dir), "rev-parse", "--verify", f"refs/heads/{branch}"]
            )
            if rc != 0:
                LOG.info("Target local branch %s not present; attempting remote branch checkout.", branch)
                rc, out = _run_cmd(
                    [
                        "git",
                        "-C",
                        str(repo_dir),
                        "checkout",
                        "-B",
                        branch,
                        f"origin/{branch}",
                    ]
                )
                if rc != 0:
                    LOG.warning("Remote branch checkout failed:\n%s", out.strip())
                    LOG.info("Falling back to fresh clone.")
                    shutil.rmtree(repo_dir)
            else:
                rc, out = _run_cmd(["git", "-C", str(repo_dir), "checkout", branch])
                if rc != 0:
                    LOG.warning("Failed to checkout branch %s:\n%s", branch, out.strip())
                else:
                    rc, out = _run_cmd(["git", "-C", str(repo_dir), "pull", "--ff-only", "origin", branch])
                    if rc != 0:
                        LOG.warning("git pull failed for %s:\n%s", branch, out.strip())
                        LOG.info("Falling back to fresh clone.")
                        shutil.rmtree(repo_dir)
        else:
            rc, out = _run_cmd(["git", "-C", str(repo_dir), "pull", "--ff-only"])
            if rc != 0:
                LOG.warning("git pull failed:\n%s", out.strip())
                LOG.info("Falling back to fresh clone.")
                shutil.rmtree(repo_dir)

    if not repo_dir.exists():
        clone_cmd = ["git", "clone"]
        if branch:
            clone_cmd.extend(["--branch", branch])
        clone_cmd.extend([repo_url, str(repo_dir)])
        rc, out = _run_cmd(clone_cmd)
        if rc != 0:
            raise RuntimeError(f"Failed to clone Echoverse repo ({repo_url}).\n{out}")


def _db_exists_for_env(env_dir: Path) -> bool:
    # DB files may be directly under env dir or nested in task-specific folders.
    return any(p.suffix == ".db" for p in env_dir.rglob("*.db"))


def ensure_env_dbs(repo_dir: Path, dataset: str, requested_envs: List[str]) -> None:
    envs_root = repo_dir / "envs"
    if not envs_root.exists():
        raise FileNotFoundError(f"Echoverse envs directory not found: {envs_root}")

    missing: List[str] = []
    for env_name in requested_envs:
        env_dir = envs_root / env_name
        if not _db_exists_for_env(env_dir):
            missing.append(env_name)
    if not missing:
        LOG.info("Grounding DBs are already present for requested environments.")
        return

    LOG.info(
        "Missing DB assets for environments: %s. Downloading from Hugging Face: %s",
        ", ".join(missing),
        dataset,
    )
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    env = os.environ.copy()
    if hf_token:
        env["HF_TOKEN"] = hf_token
    include_patterns: List[str] = []
    for env_name in missing:
        include_patterns.append(f"{env_name}/*.db")
        include_patterns.append(f"{env_name}/**/*.db")
    cmd = [
        "hf",
        "download",
        dataset,
        "--repo-type",
        "dataset",
        "--local-dir",
        str(envs_root),
    ]
    for pattern in include_patterns:
        cmd.extend(["--include", pattern])
    rc, out = _run_cmd(cmd, cwd=repo_dir, env=env)
    if rc != 0:
        # Keep best effort if hf CLI not available; don't silently continue with
        # missing files.
        raise RuntimeError(
            "Failed to download Echoverse grounding DBs with `hf download`.\n"
            "Install with: pip install -U huggingface_hub\n"
            f"{out}"
        )

    # Post-check.
    still_missing = [e for e in missing if not _db_exists_for_env(envs_root / e)]
    if still_missing:
        raise RuntimeError(
            f"HF download completed, but DB files still missing for: {', '.join(still_missing)}"
        )


def load_env_tasks(repo_dir: Path, env_name: str) -> List[Dict[str, Any]]:
    task_file = repo_dir / "envs" / env_name / "tasks" / "test_tasks.jsonl"
    if not task_file.exists():
        raise FileNotFoundError(f"Task file not found: {task_file}")

    tasks: List[Dict[str, Any]] = []
    for line in task_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            tasks.append(json.loads(line))
        except Exception:
            continue
    return tasks


def parse_score_from_verify_output(text: str) -> Optional[float]:
    patterns = (
        r"score\s*=\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)",
        r"score\s*:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)",
    )
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                continue
    return None


def parse_verdict_from_verify_output(text: str) -> Optional[str]:
    m = re.search(r"\b(PASS|FAIL)\b.*\bscore\b", text, re.IGNORECASE | re.DOTALL)
    if not m:
        return None
    return m.group(1).upper()


def read_duration_seconds(trajectory_dir: Path) -> Optional[float]:
    times_file = trajectory_dir / "times.json"
    if not times_file.exists():
        return None
    try:
        payload = json.loads(times_file.read_text(encoding="utf-8"))
        duration = payload.get("duration")
        if isinstance(duration, (int, float)):
            return float(duration)
    except Exception:
        return None
    return None


def _find_run_root(
    output_root: Path,
    env_name: str,
    agent: str,
    user: str,
    run_id: str,
) -> Path:
    # Primary path used by the harness docs and examples.
    candidate_roots = [
        output_root / "SyntheticEnv-eval" / f"{env_name}_{agent}" / user / "SyntheticEnv" / run_id,
        output_root / "SyntheticEnv-eval" / f"{env_name}_{agent}" / user / run_id,
        output_root / f"{env_name}_{agent}" / user / "SyntheticEnv" / run_id,
    ]

    for candidate in candidate_roots:
        if candidate.exists():
            return candidate

    # Fallback: find any directory containing this run_id under output_root.
    matches = [p for p in output_root.rglob(f"*{run_id}*") if p.is_dir() and p.name == run_id]
    if matches:
        # stable, deterministic fallback
        return sorted(matches)[0]
    return candidate_roots[0]


def _resolve_trajectory_dir(run_root: Path, output_root: Path, run_id: str, task_id: str) -> Path:
    direct = run_root / task_id
    if direct.exists():
        return direct

    # Fallback if harness layout diverges.
    matches = list((output_root.rglob(f"*{run_id}*")) )
    for p in matches:
        cand = p / task_id
        if cand.exists():
            return cand
    return direct


def batch_run(
    repo_dir: Path,
    env_name: str,
    task_ids: List[str],
    seed: int,
    output_root: Path,
    args: argparse.Namespace,
    run_id: str,
    user: str,
) -> Tuple[int, float, str, Path]:
    run_root = _find_run_root(output_root, env_name, args.agent, user, run_id)
    run_root.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        "-m",
        "harness.eval.batch",
        "--env",
        env_name,
        "--agent",
        args.agent,
        "--num-tasks",
        str(len(task_ids)),
        "--seed",
        str(seed),
        "--output-root",
        str(output_root),
        "--run-id",
        run_id,
        "--user",
        user,
        "--agent-model",
        args.agent_model,
        "--agent-api-key",
        args.agent_api_key,
        "--max-rounds",
        str(args.max_rounds),
        "--system",
        "SyntheticEnv-eval",
        "--experiment",
        f"{env_name}_{args.agent}",
    ]

    if args.agent_base_url:
        cmd.extend(["--agent-base-url", args.agent_base_url])
    if args.agent_base_urls:
        cmd.extend(["--agent-base-urls", ",".join(args.agent_base_urls)])
    if args.concurrency is not None:
        cmd.extend(["--concurrency", str(args.concurrency)])
    if args.headful:
        cmd.append("--headful")

    start = time.time()
    LOG.info("Running harness batch: env=%s seed=%s task_count=%d", env_name, seed, len(task_ids))
    rc, out = _run_cmd(cmd, cwd=repo_dir)
    runtime = time.time() - start
    if rc != 0:
        LOG.error("Batch run failed for env=%s seed=%s:\n%s", env_name, seed, out.strip())
    return rc, runtime, out, run_root


def verify_task(
    repo_dir: Path,
    env_name: str,
    seed: int,
    task_id: str,
    final_db: Path,
    args: argparse.Namespace,
) -> TaskResult:
    if not final_db.exists():
        return TaskResult(
            env=env_name,
            seed=seed,
            task_id=task_id,
            score=None,
            pass_fail=None,
            reason=f"Missing final_db: {final_db}",
            duration_s=None,
            verify_rc=2,
            final_db=str(final_db),
        )

    cmd = [
        "python",
        "-m",
        "harness.verify_cli",
        "--env",
        env_name,
        "--task",
        task_id,
        "--final-db",
        str(final_db),
        "--auth",
        args.judge_auth,
    ]
    if args.judge_model:
        cmd.extend(["--model", args.judge_model])
    if args.judge_base_url:
        cmd.extend(["--base-url", args.judge_base_url])

    rc, out = _run_cmd(cmd, cwd=repo_dir)
    score = parse_score_from_verify_output(out)
    verdict = parse_verdict_from_verify_output(out)
    if verdict is not None:
        reason = verdict
    elif rc == 0:
        reason = "UNSCORED_SUCCESS"
    else:
        reason = "VERIFY_FAILED"

    return TaskResult(
        env=env_name,
        seed=seed,
        task_id=task_id,
        score=score,
        pass_fail=verdict,
        reason=reason,
        duration_s=None,
        verify_rc=rc,
        final_db=str(final_db),
        output=_shorten(out),
    )


def verify_run(
    repo_dir: Path,
    env_name: str,
    seed: int,
    task_ids: List[str],
    run_id: str,
    run_root: Path,
    output_root: Path,
    args: argparse.Namespace,
) -> List[TaskResult]:
    results: List[TaskResult] = []
    for task_id in task_ids:
        traj_dir = _resolve_trajectory_dir(run_root, output_root, run_id, task_id)
        final_db = traj_dir / "final_db_state.db"
        result = verify_task(
            repo_dir=repo_dir,
            env_name=env_name,
            seed=seed,
            task_id=task_id,
            final_db=final_db,
            args=args,
        )
        if result.duration_s is None:
            result.duration_s = read_duration_seconds(traj_dir)
        results.append(result)
    return results


def summarize_and_log_wandb(results: List[RunResult], args: argparse.Namespace) -> None:
    if not args.use_wandb:
        return
    try:
        import wandb
    except Exception as exc:
        LOG.warning("wandb unavailable: %s", exc)
        return

    run_name = f"{args.run_name_prefix}-{int(time.time())}"
    config = vars(args).copy()
    config["env_count"] = len({(r.env, r.seed) for r in results})
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=run_name,
        config=config,
    )

    for i, rr in enumerate(results):
        step = rr.seed * 1000 + i
        wandb.log(
            {
                "run_id": rr.run_id,
                "env": rr.env,
                "seed": rr.seed,
                "batch_rc": rr.batch_rc,
                "batch_runtime_s": rr.batch_runtime_s,
                "tasks_requested": rr.tasks_requested,
                "task_count": rr.task_count,
                "verified": rr.verified,
                "passed": rr.passed,
                "failed": rr.failed,
                "errors": rr.errors,
                "pass_rate": rr.pass_rate,
            },
            step=step,
        )

        if rr.results:
            # avoid logging huge nested structures as a single row
            total = rr.task_count or 1
            wandb.log(
                {
                    "pass_rate_over_tasks": rr.passed / total,
                    "verified_rate": rr.verified / total,
                },
                step=step,
            )

    run_pass_rates = [r.pass_rate for r in results if r.verified > 0]
    if run_pass_rates:
        wandb.summary["overall_mean_pass_rate"] = mean(run_pass_rates)
    wandb.summary["runs"] = len(results)
    wandb.summary["successful_runs"] = sum(1 for r in results if r.batch_rc == 0)
    if wandb.run is not None:
        wandb.finish()


def write_outputs(output_root: Path, results: List[RunResult]) -> None:
    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runs": [r.as_json() for r in results],
    }
    manifest_path = output_root / "echoverse_matrix_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    LOG.info("Wrote run manifest: %s", manifest_path)

    csv_path = output_root / "echoverse_matrix_task_results.csv"
    rows: List[Dict[str, Any]] = []
    for rr in results:
        for tr in rr.results:
            rows.append(
                {
                    "env": tr.env,
                    "seed": tr.seed,
                    "task_id": tr.task_id,
                    "score": tr.score,
                    "pass_fail": tr.pass_fail,
                    "reason": tr.reason,
                    "duration_s": tr.duration_s,
                    "verify_rc": tr.verify_rc,
                    "final_db": tr.final_db,
                }
            )

    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        LOG.info("Wrote task CSV: %s", csv_path)
    else:
        LOG.warning("No task rows to write; CSV output skipped.")


def select_task_ids(tasks: List[Dict[str, Any]], args: argparse.Namespace) -> List[str]:
    task_ids = [t.get("id") for t in tasks if t.get("id")]
    task_ids = [str(t) for t in task_ids if t]

    if args.shuffle_tasks:
        rng = random.Random(args.shuffle_seed)
        rng.shuffle(task_ids)

    if args.task_limit is not None and args.task_limit > 0:
        task_ids = task_ids[: args.task_limit]

    return task_ids


def run_matrix(args: argparse.Namespace) -> List[RunResult]:
    repo_dir = Path(args.repo_dir).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    ensure_repo(repo_dir, args.repo_url, args.repo_branch)

    envs_root = repo_dir / "envs"
    if not envs_root.exists():
        raise FileNotFoundError(f"Echoverse envs directory not found: {envs_root}")

    requested_envs = [e.strip() for e in args.envs.split(",") if e.strip()]
    selected_envs = [e for e in requested_envs if e in EMBEDDED_ENVS] if requested_envs else EMBEDDED_ENVS
    ignored = [e for e in requested_envs if e not in EMBEDDED_ENVS]
    if ignored:
        LOG.warning("Ignoring unsupported env names: %s", ",".join(ignored))
    if not selected_envs:
        raise ValueError("No valid Echoverse environments selected.")

    ensure_env_dbs(repo_dir, args.hf_dataset, selected_envs)

    if args.seed_filter is not None:
        requested = set(_parse_csv_ints(args.seed_filter))
        available = _parse_csv_ints(args.seeds)
        if requested:
            available = [s for s in available if s in requested]
    else:
        available = _parse_csv_ints(args.seeds)
    seeds = available if available else [42]

    user = args.user or os.environ.get("USER", "colab")
    os.environ["ENVS_ROOT"] = str(envs_root)

    results: List[RunResult] = []
    for env_name in selected_envs:
        tasks = load_env_tasks(repo_dir, env_name)
        if not tasks:
            LOG.warning("No tasks found for env=%s; skipping.", env_name)
            continue

        base_task_ids = select_task_ids(tasks, args)
        for seed in seeds:
            task_ids = base_task_ids
            if args.limit_per_env_seed and len(task_ids) > args.limit_per_env_seed:
                task_ids = task_ids[: args.limit_per_env_seed]

            run_id = f"{env_name}_seed{seed}_n{len(task_ids)}"
            rc, batch_runtime, batch_output, run_root = batch_run(
                repo_dir=repo_dir,
                env_name=env_name,
                task_ids=task_ids,
                seed=seed,
                output_root=output_root,
                args=args,
                run_id=run_id,
                user=user,
            )

            tr_results: List[TaskResult] = []
            if rc == 0 and not args.skip_verify:
                tr_results = verify_run(
                    repo_dir=repo_dir,
                    env_name=env_name,
                    seed=seed,
                    task_ids=task_ids,
                    run_id=run_id,
                    run_root=run_root,
                    output_root=output_root,
                    args=args,
                )

            verified = sum(1 for t in tr_results if t.verify_rc == 0)
            rr = RunResult(
                env=env_name,
                seed=seed,
                run_id=run_id,
                task_count=len(task_ids),
                tasks_requested=len(base_task_ids),
                batch_rc=rc,
                batch_runtime_s=batch_runtime,
                batch_output=batch_output.strip(),
                verified=verified,
                passed=sum(1 for t in tr_results if t.pass_fail == "PASS"),
                failed=sum(1 for t in tr_results if t.pass_fail == "FAIL"),
                errors=len(tr_results) - verified,
                results=tr_results,
            )
            results.append(rr)

    write_outputs(output_root, results)
    return results


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Echoverse environments across all tasks and seeds with W&B/HF integration."
    )
    parser.add_argument("--repo-url", default="https://github.com/microsoft/Echoverse.git")
    parser.add_argument("--repo-branch", default=None)
    parser.add_argument(
        "--repo-dir",
        default="./platform_colab/echoverse_workspace",
        help="Local path for the checked-out Echoverse repo.",
    )
    parser.add_argument(
        "--hf-dataset",
        default="microsoft/Echoverse",
        help="HF dataset containing Echoverse grounding DBs.",
    )
    parser.add_argument(
        "--envs",
        default=",".join(EMBEDDED_ENVS),
        help="Comma-separated list of environments.",
    )
    parser.add_argument(
        "--seeds",
        default="0,1,2",
        help="Comma-separated random seeds (runner-level repetitions).",
    )
    parser.add_argument(
        "--seed-filter",
        default=None,
        help="Optional comma-separated subset of --seeds to execute.",
    )
    parser.add_argument(
        "--agent",
        default="fara15",
        help="Solver agent registered in Echoverse.",
    )
    parser.add_argument("--agent-base-url", default=None)
    parser.add_argument("--agent-base-urls", default=None)
    parser.add_argument("--agent-model", default="Fara1.5-9B")
    parser.add_argument("--agent-api-key", default="not-needed")
    parser.add_argument("--max-rounds", type=int, default=120)
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--headful", action="store_true")
    parser.add_argument(
        "--task-limit",
        type=int,
        default=None,
        help="Optional upper bound on number of tasks per env (smoke test).",
    )
    parser.add_argument(
        "--limit-per-env-seed",
        type=int,
        default=None,
        help="Optional backward-compatible per-seed task truncation bound.",
    )
    parser.add_argument("--shuffle-tasks", action="store_true")
    parser.add_argument("--shuffle-seed", type=int, default=0)
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--judge-base-url", default=None)
    parser.add_argument(
        "--judge-auth",
        default="auto",
        choices=["auto", "openai", "azure-key", "azure-ad"],
    )
    parser.add_argument("--output-root", default="./platform_colab/echoverse_outputs")
    parser.add_argument("--run-name-prefix", default="echoverse-matrix")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="tinker-echoverse")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--user", default=None)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    args.agent_base_urls = (
        [u.strip() for u in args.agent_base_urls.split(",") if u.strip()]
        if args.agent_base_urls
        else None
    )
    if args.agent_base_urls and args.agent_base_url:
        raise ValueError(
            "Provide either --agent-base-url or --agent-base-urls, not both."
        )

    return args


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if args.dry_run:
        LOG.info("DRY-RUN requested. Planned command arguments:")
        print(json.dumps(vars(args), indent=2))
        return 0

    results = run_matrix(args)
    summarize_and_log_wandb(results, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
