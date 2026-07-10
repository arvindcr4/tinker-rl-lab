#!/usr/bin/env python3
"""Deterministic, dependency-free MTech defense demo for TinkerRL.

The default path is deliberately offline. It computes group-relative advantages
on an explicitly synthetic fixture and audits a recorded project artifact for
byte provenance and internal arithmetic. Neither check is a new scientific
result. The optional live mode is only an endpoint/schema connectivity smoke.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import subprocess
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from socketserver import TCPServer
from typing import Any, Dict, List, Optional, Sequence, Tuple


DEMO_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEMO_DIR.parents[1]
DEFAULT_FIXTURE = DEMO_DIR / "fixtures" / "offline_demo.json"
DEFAULT_OUTPUT_DIR = DEMO_DIR / "output"
GROQ_MODEL = "kimi-k2-0905-preview"
EPSILON = 1e-12


class DemoError(RuntimeError):
    """Expected input, integrity, or live-mode failure."""


def load_json(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DemoError("required file not found: {}".format(path)) from exc
    except json.JSONDecodeError as exc:
        raise DemoError("invalid JSON in {}: {}".format(path, exc)) from exc
    if not isinstance(data, dict):
        raise DemoError("expected a JSON object in {}".format(path))
    return data


def _numeric_rewards(raw: Any, group_id: str) -> List[float]:
    if not isinstance(raw, list) or len(raw) < 2:
        raise DemoError("group {} must contain at least two rewards".format(group_id))
    rewards: List[float] = []
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise DemoError("group {} contains a non-numeric reward".format(group_id))
        value_float = float(value)
        if not math.isfinite(value_float):
            raise DemoError("group {} contains a non-finite reward".format(group_id))
        rewards.append(value_float)
    return rewards


def compute_group(rewards: Sequence[float]) -> Dict[str, Any]:
    """Return population statistics and GRPO-style normalized advantages."""
    if len(rewards) < 2:
        raise DemoError("a group needs at least two rewards")
    values = [float(value) for value in rewards]
    mean_reward = sum(values) / len(values)
    variance = sum((value - mean_reward) ** 2 for value in values) / len(values)
    std_reward = math.sqrt(variance)
    zero_variance = std_reward <= EPSILON
    if zero_variance:
        advantages = [0.0 for _ in values]
    else:
        advantages = [(value - mean_reward) / std_reward for value in values]
    return {
        "rewards": values,
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "zero_variance": zero_variance,
        "advantages": advantages,
    }


def analyze_fixture(fixture: Dict[str, Any]) -> Dict[str, Any]:
    groups_raw = fixture.get("groups")
    if not isinstance(groups_raw, list) or not groups_raw:
        raise DemoError("fixture must contain a non-empty groups list")

    groups: List[Dict[str, Any]] = []
    for index, raw in enumerate(groups_raw):
        if not isinstance(raw, dict):
            raise DemoError("fixture group {} is not an object".format(index))
        group_id = str(raw.get("id", "group-{}".format(index)))
        stats = compute_group(_numeric_rewards(raw.get("rewards"), group_id))
        stats.update(
            {
                "id": group_id,
                "label": str(raw.get("label", group_id)),
                "interpretation": str(raw.get("interpretation", "")),
            }
        )
        groups.append(stats)

    zero_variance_groups = sum(1 for group in groups if group["zero_variance"])
    zvf = zero_variance_groups / len(groups)
    summary = {
        "status": "PASS",
        "group_count": len(groups),
        "effective_groups": len(groups) - zero_variance_groups,
        "zero_variance_groups": zero_variance_groups,
        "zvf": zvf,
        "gradient_utilization": 1.0 - zvf,
        "groups": groups,
        "scope_notice": str(fixture.get("scope_notice", "")),
    }

    expected = fixture.get("expected", {})
    for key in ("zvf", "gradient_utilization"):
        if key in expected and not math.isclose(
            float(summary[key]), float(expected[key]), rel_tol=0.0, abs_tol=EPSILON
        ):
            raise DemoError(
                "fixture contract mismatch for {}: computed {}, expected {}".format(
                    key, summary[key], expected[key]
                )
            )
    if "effective_groups" in expected and summary["effective_groups"] != int(
        expected["effective_groups"]
    ):
        raise DemoError("fixture contract mismatch for effective_groups")
    return summary


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_recorded_artifact(path: Path, contract: Dict[str, Any]) -> Dict[str, Any]:
    """Check exact bytes and recompute all aggregates in a recorded result."""
    data = load_json(path)
    actual_sha256 = sha256_file(path)
    expected_sha256 = str(contract.get("sha256", ""))
    if actual_sha256 != expected_sha256:
        raise DemoError(
            "artifact SHA-256 mismatch: got {}, expected {}".format(
                actual_sha256, expected_sha256
            )
        )

    rows = data.get("per_problem")
    if not isinstance(rows, list) or not rows:
        raise DemoError("recorded artifact has no per_problem rows")
    if int(data.get("n_problems", -1)) != len(rows):
        raise DemoError("n_problems does not match per_problem row count")
    if len(rows) != int(contract.get("n_problems", -1)):
        raise DemoError("artifact row count does not match the demo contract")
    if str(data.get("model")) != str(contract.get("model_label")):
        raise DemoError("artifact model label does not match the demo contract")

    group_size = int(contract.get("group_size", -1))
    all_rewards: List[float] = []
    recomputed_zvf: List[float] = []
    checked_rows: List[Dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise DemoError("artifact row {} is not an object".format(index))
        rewards = _numeric_rewards(row.get("rewards"), str(row.get("problem_id", index)))
        if len(rewards) != group_size:
            raise DemoError("artifact row {} has an unexpected group size".format(index))
        stats = compute_group(rewards)
        row_zvf = 1.0 if stats["zero_variance"] else 0.0
        if not math.isclose(
            stats["mean_reward"], float(row.get("mean_reward", math.nan)), abs_tol=EPSILON
        ):
            raise DemoError("artifact row {} mean_reward mismatch".format(index))
        if not math.isclose(row_zvf, float(row.get("zvf", math.nan)), abs_tol=EPSILON):
            raise DemoError("artifact row {} ZVF mismatch".format(index))
        all_rewards.extend(rewards)
        recomputed_zvf.append(row_zvf)
        checked_rows.append(
            {
                "problem_id": row.get("problem_id", index),
                "mean_reward": stats["mean_reward"],
                "zvf": row_zvf,
            }
        )

    overall_accuracy = sum(all_rewards) / len(all_rewards)
    overall_zvf = sum(recomputed_zvf) / len(recomputed_zvf)
    expected_accuracy = float(contract.get("overall_accuracy", math.nan))
    expected_zvf = float(contract.get("overall_zvf", math.nan))
    for label, computed, recorded, expected in (
        ("overall_accuracy", overall_accuracy, float(data.get("overall_accuracy", math.nan)), expected_accuracy),
        ("overall_zvf", overall_zvf, float(data.get("overall_zvf", math.nan)), expected_zvf),
    ):
        if not math.isclose(computed, recorded, abs_tol=EPSILON):
            raise DemoError("artifact {} arithmetic mismatch".format(label))
        if not math.isclose(computed, expected, abs_tol=EPSILON):
            raise DemoError("artifact {} contract mismatch".format(label))

    return {
        "status": "PASS",
        "path": str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path),
        "sha256": actual_sha256,
        "model_label": str(data.get("model")),
        "n_problems": len(rows),
        "group_size": group_size,
        "reward_count": len(all_rewards),
        "overall_accuracy": overall_accuracy,
        "overall_zvf": overall_zvf,
        "checked_rows": checked_rows,
        "scope_notice": str(contract.get("scope_notice", "")),
    }


def _extract_json_object(text: str) -> Dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise DemoError("live response did not contain a JSON object")


def run_live_smoke(tasks: Any) -> Dict[str, Any]:
    """Use the repository-mandated Groq/Kimi path for a tiny schema smoke."""
    if not isinstance(tasks, list) or not tasks:
        raise DemoError("fixture does not define live_smoke_tasks")
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise DemoError("live mode requires GROQ_API_KEY")
    try:
        from groq import Groq
    except ImportError as exc:
        raise DemoError("live mode requires the groq package: pip install groq") from exc

    client = Groq(api_key=api_key)
    rows: List[Dict[str, Any]] = []
    for task in tasks:
        completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a deterministic endpoint smoke test. Return only one JSON object "
                        "with an integer field named answer. Do not add markdown or explanation."
                    ),
                },
                {"role": "user", "content": str(task["prompt"])},
            ],
            model=GROQ_MODEL,
            temperature=0,
        )
        content = completion.choices[0].message.content or ""
        parsed = _extract_json_object(content)
        answer = parsed.get("answer")
        expected = int(task["expected_answer"])
        passed = isinstance(answer, int) and not isinstance(answer, bool) and answer == expected
        rows.append(
            {
                "id": str(task["id"]),
                "expected_answer": expected,
                "parsed_answer": answer,
                "schema_and_answer_pass": passed,
            }
        )
    passed_count = sum(1 for row in rows if row["schema_and_answer_pass"])
    return {
        "status": "PASS" if passed_count == len(rows) else "FAIL",
        "provider": "Groq",
        "model": GROQ_MODEL,
        "passed": passed_count,
        "total": len(rows),
        "rows": rows,
        "scope_notice": (
            "Endpoint and JSON-schema smoke over fixed toy arithmetic only. "
            "It is not a benchmark, training evaluation, or model-quality claim."
        ),
    }


def _fmt_number(value: float) -> str:
    if math.isclose(value, round(value), abs_tol=EPSILON):
        return str(int(round(value)))
    return "{:.3f}".format(value).rstrip("0").rstrip(".")


def _render_group_rows(groups: Sequence[Dict[str, Any]]) -> str:
    rows = []
    for group in groups:
        rewards = ", ".join(_fmt_number(value) for value in group["rewards"])
        advantages = ", ".join("{:+.3f}".format(value) for value in group["advantages"])
        signal = "No contrast" if group["zero_variance"] else "Contrast available"
        signal_class = "warn" if group["zero_variance"] else "pass"
        rows.append(
            "<tr><td><strong>{}</strong><br><small>{}</small></td>"
            "<td><code>[{}]</code></td><td>{:.3f}</td><td>{:.3f}</td>"
            "<td><code>[{}]</code></td><td><span class='badge {}'>{}</span></td></tr>".format(
                html.escape(str(group["label"])),
                html.escape(str(group["interpretation"])),
                html.escape(rewards),
                group["mean_reward"],
                group["std_reward"],
                html.escape(advantages),
                signal_class,
                signal,
            )
        )
    return "\n".join(rows)


def render_html(report: Dict[str, Any]) -> str:
    mechanism = report["mechanism"]
    artifact = report["artifact_audit"]
    live = report.get("live_smoke")
    live_html = ""
    if live:
        live_html = """
        <section>
          <h2>Optional live endpoint smoke</h2>
          <div class="metric"><span>{passed}/{total}</span><small>toy schema checks</small></div>
          <p>{notice}</p>
        </section>
        """.format(
            passed=live["passed"], total=live["total"], notice=html.escape(live["scope_notice"])
        )

    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TinkerRL MTech Defense Demo</title>
  <style>
    :root {{ --ink:#172033; --muted:#5f6b7a; --panel:#fff; --line:#d8dee9;
      --navy:#153d6f; --cyan:#007f8b; --green:#147d4f; --amber:#a65b00; --bg:#f4f7fb; }}
    * {{ box-sizing:border-box; }} body {{ margin:0; color:var(--ink); background:var(--bg);
      font:16px/1.45 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    header {{ padding:42px clamp(20px,6vw,76px); color:white;
      background:linear-gradient(120deg,var(--navy),#245d8c 62%,var(--cyan)); }}
    header p {{ max-width:850px; margin:.75rem 0 0; color:#e6f5ff; }}
    main {{ max-width:1200px; margin:0 auto; padding:26px 20px 50px; }}
    section {{ background:var(--panel); border:1px solid var(--line); border-radius:14px;
      padding:24px; margin:18px 0; box-shadow:0 8px 24px rgba(30,55,85,.06); }}
    h1 {{ margin:0; font-size:clamp(30px,5vw,52px); line-height:1.08; }} h2 {{ margin-top:0; }}
    .eyebrow {{ text-transform:uppercase; letter-spacing:.14em; font-weight:700; font-size:12px; }}
    .metrics {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:14px; }}
    .metric {{ border-left:5px solid var(--cyan); background:#edf7fa; border-radius:8px; padding:16px; }}
    .metric span {{ display:block; font-size:30px; font-weight:800; }} .metric small {{ color:var(--muted); }}
    .badge {{ display:inline-block; border-radius:999px; padding:3px 9px; font-size:12px; font-weight:800; }}
    .badge.pass {{ color:var(--green); background:#e3f5ec; }} .badge.warn {{ color:var(--amber); background:#fff0d8; }}
    .scope {{ border-left:5px solid var(--amber); background:#fff8eb; padding:14px 16px; border-radius:8px; }}
    .two {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:16px; }}
    table {{ width:100%; border-collapse:collapse; font-size:14px; }} th,td {{ text-align:left; padding:11px 9px;
      border-bottom:1px solid var(--line); vertical-align:top; }} th {{ color:var(--muted); font-size:12px; text-transform:uppercase; }}
    code {{ white-space:nowrap; }} .table-wrap {{ overflow-x:auto; }} .hash {{ word-break:break-all; font-family:ui-monospace,monospace; }}
    footer {{ color:var(--muted); text-align:center; padding:18px; font-size:13px; }}
  </style>
</head>
<body>
<header>
  <div class="eyebrow">PES MTech thesis defense · deterministic offline mode</div>
  <h1>TinkerRL: group-relative signal audit</h1>
  <p>A compact demonstration of how within-group rewards become normalized advantages, why equal-reward groups provide no group-relative contrast, and how a recorded result artifact can be checked without a GPU or network.</p>
</header>
<main>
  <section>
    <h2>Run status</h2>
    <div class="metrics">
      <div class="metric"><span>{group_count}</span><small>synthetic groups</small></div>
      <div class="metric"><span>{zvf:.0%}</span><small>zero-variance fraction (fixture)</small></div>
      <div class="metric"><span>{gu:.0%}</span><small>gradient utilization proxy (fixture)</small></div>
      <div class="metric"><span>{artifact_status}</span><small>recorded artifact integrity</small></div>
    </div>
    <p class="scope"><strong>Scope boundary:</strong> {mechanism_notice}</p>
  </section>
  <section>
    <h2>1. Mechanism: rewards → relative advantages</h2>
    <p>For each prompt group, the demo computes <code>Aᵢ = (rᵢ − mean(r)) / std(r)</code>. When the population standard deviation is zero, every displayed advantage is set to zero.</p>
    <div class="table-wrap"><table>
      <thead><tr><th>Group</th><th>Rewards</th><th>Mean</th><th>Std.</th><th>Advantages</th><th>Signal</th></tr></thead>
      <tbody>{group_rows}</tbody>
    </table></div>
  </section>
  <section>
    <h2>2. Recorded artifact audit</h2>
    <div class="two">
      <div><p><strong>Input</strong><br><code>{artifact_path}</code></p>
        <p><strong>SHA-256</strong><br><span class="hash">{artifact_hash}</span></p></div>
      <div class="metrics">
        <div class="metric"><span>{reward_count}</span><small>recorded binary rewards rechecked</small></div>
        <div class="metric"><span>{artifact_accuracy:.2%}</span><small>recomputed recorded reward mean</small></div>
        <div class="metric"><span>{artifact_zvf:.2%}</span><small>recomputed recorded ZVF</small></div>
      </div>
    </div>
    <p class="scope"><strong>What this proves:</strong> exact input bytes and internal aggregate arithmetic agree. <strong>What it does not prove:</strong> {artifact_notice}</p>
  </section>
  <section class="two">
    <div><h2>What the evaluator can verify</h2><ul>
      <li>Default run needs only Python 3.9+ standard library.</li>
      <li>The synthetic fixture has an explicit expected-value contract.</li>
      <li>Every recorded per-problem mean and ZVF is recomputed.</li>
      <li>JSON and HTML outputs are deterministic byte-for-byte.</li>
    </ul></div>
    <div><h2>Claims intentionally excluded</h2><ul>
      <li>No claim that this synthetic fixture is model training.</li>
      <li>No claim that online reward is held-out accuracy.</li>
      <li>No claim of causal improvement or state of the art.</li>
      <li>No live service is required for the defense path.</li>
    </ul></div>
  </section>
  {live_html}
</main>
<footer>Generated locally by <code>submission/demo/demo.sh</code> · no telemetry · no external assets</footer>
</body></html>
""".format(
        group_count=mechanism["group_count"],
        zvf=mechanism["zvf"],
        gu=mechanism["gradient_utilization"],
        artifact_status=html.escape(str(artifact["status"])),
        mechanism_notice=html.escape(mechanism["scope_notice"]),
        group_rows=_render_group_rows(mechanism["groups"]),
        artifact_path=html.escape(artifact["path"]),
        artifact_hash=html.escape(artifact["sha256"]),
        reward_count=artifact["reward_count"],
        artifact_accuracy=artifact["overall_accuracy"],
        artifact_zvf=artifact["overall_zvf"],
        artifact_notice=html.escape(artifact["scope_notice"]),
        live_html=live_html,
    )


def write_report(report: Dict[str, Any], output_dir: Path) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "demo_report.json"
    html_path = output_dir / "demo_report.html"
    json_text = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    json_path.write_text(json_text, encoding="utf-8")
    html_path.write_text(render_html(report), encoding="utf-8")
    return json_path, html_path


def serve(output_dir: Path, port: int) -> None:
    class LocalHTTPServer(ThreadingHTTPServer):
        def server_bind(self) -> None:
            # http.server.HTTPServer performs a reverse-DNS lookup here. That
            # can hang on locked-down or offline defense networks, so bind via
            # TCPServer and use the known loopback name directly.
            TCPServer.server_bind(self)
            self.server_name = "127.0.0.1"
            self.server_port = self.server_address[1]

    class QuietHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, directory=str(output_dir), **kwargs)

        def log_message(self, format_string: str, *args: Any) -> None:
            return

    server = LocalHTTPServer(("127.0.0.1", port), QuietHandler)
    print("Dashboard: http://127.0.0.1:{}/demo_report.html".format(port), flush=True)
    print("Press Ctrl-C to stop the local server.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nLocal server stopped.")
    finally:
        server.server_close()


def run_self_tests() -> int:
    command = [
        sys.executable,
        "-m",
        "unittest",
        "discover",
        "-s",
        str(DEMO_DIR / "tests"),
        "-v",
    ]
    return subprocess.run(command, cwd=str(REPO_ROOT), check=False).returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the deterministic TinkerRL MTech defense demo."
    )
    parser.add_argument("--mode", choices=("offline", "live"), default="offline")
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--artifact", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--serve", action="store_true", help="serve the generated dashboard locally")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--self-test", action="store_true", help="run the stdlib smoke-test suite")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        return run_self_tests()

    try:
        fixture = load_json(args.fixture.resolve())
        mechanism = analyze_fixture(fixture)
        contract = fixture.get("artifact_contract")
        if not isinstance(contract, dict):
            raise DemoError("fixture does not contain an artifact_contract")
        artifact_path = (
            args.artifact.resolve()
            if args.artifact
            else REPO_ROOT / str(contract["path_from_repo_root"])
        )
        artifact_audit = audit_recorded_artifact(artifact_path, contract)
        report: Dict[str, Any] = {
            "schema_version": 1,
            "demo_status": "PASS",
            "mode": args.mode,
            "mechanism": mechanism,
            "artifact_audit": artifact_audit,
            "claim_boundary": (
                "Mechanism and integrity demonstration only; no new scientific or model-performance claim."
            ),
        }
        if args.mode == "live":
            live = run_live_smoke(fixture.get("live_smoke_tasks"))
            report["live_smoke"] = live
            if live["status"] != "PASS":
                report["demo_status"] = "FAIL"

        json_path, html_path = write_report(report, args.output_dir.resolve())
    except (DemoError, KeyError, OSError, ValueError) as exc:
        print("DEMO STATUS: FAIL", file=sys.stderr)
        print("Reason: {}".format(exc), file=sys.stderr)
        return 1

    print("TinkerRL MTech Defense Demo")
    print("Mode: {}".format(args.mode))
    print("Mechanism fixture: PASS ({} groups, ZVF={:.3f}, GU={:.3f})".format(
        mechanism["group_count"], mechanism["zvf"], mechanism["gradient_utilization"]
    ))
    print("Recorded artifact: PASS ({} rewards, mean={:.4f}, ZVF={:.4f})".format(
        artifact_audit["reward_count"],
        artifact_audit["overall_accuracy"],
        artifact_audit["overall_zvf"],
    ))
    print("Artifact SHA-256: {}".format(artifact_audit["sha256"]))
    if report.get("live_smoke"):
        live = report["live_smoke"]
        print("Live Groq/Kimi smoke: {} ({}/{})".format(live["status"], live["passed"], live["total"]))
    print("JSON report: {}".format(json_path))
    print("HTML dashboard: {}".format(html_path))
    print("Claim boundary: {}".format(report["claim_boundary"]))
    print("DEMO STATUS: {}".format(report["demo_status"]))

    if report["demo_status"] != "PASS":
        return 1
    if args.serve:
        try:
            serve(args.output_dir.resolve(), args.port)
        except OSError as exc:
            print("LOCAL SERVER STATUS: FAIL", file=sys.stderr)
            print("Reason: {}".format(exc), file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
