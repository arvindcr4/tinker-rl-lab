#!/usr/bin/env python3
"""Offline, stdlib-only defense evidence checks and dashboard generator."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import zipfile
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET


PACKAGE_DIR = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
OUTPUT_DIR = PACKAGE_DIR / "output"
EPS = 1e-9
NS = {
    "m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "pr": "http://schemas.openxmlformats.org/package/2006/relationships",
}


class CheckError(RuntimeError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckError(f"cannot load {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CheckError(f"expected JSON object in {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_manifest() -> dict[str, Any]:
    manifest = load_json(DATA_DIR / "manifest.json")
    checked = []
    for relative, expected in manifest["files"].items():
        path = (DATA_DIR / relative).resolve()
        if not path.is_file():
            raise CheckError(f"manifest file missing: {relative}")
        actual = sha256(path)
        if actual != expected:
            raise CheckError(f"SHA-256 mismatch for {relative}")
        checked.append({"path": relative, "sha256": actual})
    return {"status": "PASS", "file_count": len(checked), "files": checked}


def check_synthetic_fixture() -> dict[str, Any]:
    fixture = load_json(DATA_DIR / "offline_demo.json")
    groups = fixture.get("groups")
    if not isinstance(groups, list) or not groups:
        raise CheckError("offline_demo.json has no groups")
    zero_variance = 0
    rows = []
    for group in groups:
        rewards = [float(value) for value in group["rewards"]]
        mean = sum(rewards) / len(rewards)
        variance = sum((value - mean) ** 2 for value in rewards) / len(rewards)
        is_zero = variance <= EPS
        zero_variance += int(is_zero)
        rows.append({"id": group["id"], "mean": mean, "zero_variance": is_zero})
    zvf = zero_variance / len(groups)
    utilization = 1.0 - zvf
    expected = fixture["expected"]
    if not math.isclose(zvf, float(expected["zvf"]), abs_tol=EPS):
        raise CheckError("synthetic ZVF contract failed")
    if not math.isclose(utilization, float(expected["gradient_utilization"]), abs_tol=EPS):
        raise CheckError("synthetic utilization contract failed")
    return {
        "status": "PASS",
        "groups": rows,
        "zvf": zvf,
        "gradient_utilization": utilization,
        "scope_notice": fixture["scope_notice"],
    }


CLAIM2_EXPECTED = {
    "er2b_g2_s123.json": (2560, 0.900000, 0.975000),
    "er2b_g2_s456.json": (2560, 0.962500, 0.975000),
    "er2b_g16_s123.json": (2560, 0.321875, 0.150000),
    "er2b_g16_s456.json": (2560, 0.3890625, 0.100000),
}


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def check_claim2() -> dict[str, Any]:
    rows = []
    for filename, (expected_budget, expected_reward, expected_zvf) in CLAIM2_EXPECTED.items():
        data = load_json(RAW_DIR / filename)
        step_log = data.get("step_log")
        if data.get("status") != "completed" or not isinstance(step_log, list) or len(step_log) < 10:
            raise CheckError(f"incomplete Claim 2 run: {filename}")
        budget = int(data["group_size"]) * int(data["batch"]) * int(data["steps"])
        last10 = step_log[-10:]
        reward = mean([float(row["reward"]) for row in last10])
        zvf = mean([float(row["zvf"]) for row in last10])
        if budget != expected_budget:
            raise CheckError(f"matched-budget failure in {filename}: {budget}")
        if not math.isclose(reward, expected_reward, abs_tol=EPS):
            raise CheckError(f"late reward mismatch in {filename}")
        if not math.isclose(zvf, expected_zvf, abs_tol=EPS):
            raise CheckError(f"late ZVF mismatch in {filename}")
        rows.append({
            "run": filename.removesuffix(".json"),
            "group_size": int(data["group_size"]),
            "seed": int(data["seed"]),
            "steps": int(data["steps"]),
            "rollout_budget": budget,
            "late_reward_mean": reward,
            "late_zvf_mean": zvf,
            "late_zvf_min": min(float(row["zvf"]) for row in last10),
            "late_zvf_max": max(float(row["zvf"]) for row in last10),
        })
    g2 = [row for row in rows if row["group_size"] == 2]
    g16 = [row for row in rows if row["group_size"] == 16]
    return {
        "status": "PASS",
        "matched_rollout_budget": 2560,
        "rows": rows,
        "summary": {
            "g2_late_reward_range": [min(row["late_reward_mean"] for row in g2), max(row["late_reward_mean"] for row in g2)],
            "g2_late_zvf_mean_range": [min(row["late_zvf_mean"] for row in g2), max(row["late_zvf_mean"] for row in g2)],
            "g16_late_reward_range": [min(row["late_reward_mean"] for row in g16), max(row["late_reward_mean"] for row in g16)],
            "g16_late_zvf_mean_range": [min(row["late_zvf_mean"] for row in g16), max(row["late_zvf_mean"] for row in g16)],
        },
        "interpretation": "At equal 2,560-rollout budgets, G=2 receives 160 optimizer steps and reaches the all-correct high-ZVF wall; G=16 receives 20 steps and remains mid-learning. This is a two-seed trajectory observation, not a universal group-size optimum.",
    }


P4_EXPECTED = {
    "p4uncap_drgrpo_s123.json": (972.375, 902.040625, 7.233256202596737),
    "p4uncap_drgrpo_s42.json": (999.43125, 931.375, 6.809497901931722),
    "p4uncap_drgrpo_s456.json": (1000.20625, 878.23125, 12.194984784388211),
    "p4uncap_grpo_s123.json": (980.50625, 943.6125, 3.762724612923178),
    "p4uncap_grpo_s42.json": (1004.05, 904.978125, 9.867225237786961),
    "p4uncap_grpo_s456.json": (995.725, 900.090625, 9.604496723492929),
}


def check_p4() -> dict[str, Any]:
    rows = []
    for filename, expected in P4_EXPECTED.items():
        data = load_json(RAW_DIR / filename)
        step_log = data.get("step_log")
        if data.get("status") != "completed" or not isinstance(step_log, list) or len(step_log) != 30:
            raise CheckError(f"incomplete corrected P4 run: {filename}")
        first5 = mean([float(row["mean_comp_len"]) for row in step_log[:5]])
        last10 = mean([float(row["mean_comp_len"]) for row in step_log[-10:]])
        contraction = (first5 - last10) / first5 * 100.0
        for label, actual, wanted in zip(("first5", "last10", "contraction"), (first5, last10, contraction), expected):
            if not math.isclose(actual, wanted, abs_tol=EPS):
                raise CheckError(f"P4 {label} mismatch in {filename}")
        algorithm = "Dr.GRPO" if "drgrpo" in filename else "GRPO"
        rows.append({
            "run": filename.removesuffix(".json"),
            "algorithm": algorithm,
            "seed": int(data["seed"]),
            "first5_mean_length": first5,
            "last10_mean_length": last10,
            "contraction_pct": contraction,
            "late_zvf_mean": mean([float(row["zvf"]) for row in step_log[-10:]]),
        })
    contractions = [row["contraction_pct"] for row in rows]
    if round(min(contractions), 1) != 3.8 or round(max(contractions), 1) != 12.2:
        raise CheckError("P4 documented 3.8-12.2% range did not reproduce")
    return {
        "status": "PASS",
        "rows": rows,
        "exact_contraction_range_pct": [min(contractions), max(contractions)],
        "defense_wording": "Completion length contracted by approximately 3.8-12.2% in all six corrected arms (first-5 mean to last-10 mean).",
        "interpretation": "Both loss forms compress at this scale; the panel does not support a length-inflation claim. n=3 seeds per loss and 30 steps remain important limits.",
    }


def column_index(reference: str) -> int:
    letters = re.match(r"[A-Z]+", reference)
    if not letters:
        return 0
    value = 0
    for character in letters.group(0):
        value = value * 26 + ord(character) - 64
    return value - 1


def read_xlsx_sheet(path: Path, sheet_name: str) -> list[list[str]]:
    with zipfile.ZipFile(path) as archive:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall("m:si", NS):
                shared.append("".join(node.text or "" for node in item.findall(".//m:t", NS)))
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        relationship_id = None
        for sheet in workbook.findall("m:sheets/m:sheet", NS):
            if sheet.attrib.get("name") == sheet_name:
                relationship_id = sheet.attrib.get(f"{{{NS['r']}}}id")
                break
        if not relationship_id:
            raise CheckError(f"workbook sheet missing: {sheet_name}")
        rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        target = None
        for rel in rels.findall("pr:Relationship", NS):
            if rel.attrib.get("Id") == relationship_id:
                target = rel.attrib["Target"]
                break
        if not target:
            raise CheckError(f"workbook relationship missing: {sheet_name}")
        target = target.lstrip("/")
        if not target.startswith("xl/"):
            target = "xl/" + target
        root = ET.fromstring(archive.read(target))
        rows: list[list[str]] = []
        for row_node in root.findall("m:sheetData/m:row", NS):
            cells: dict[int, str] = {}
            for cell in row_node.findall("m:c", NS):
                ref = cell.attrib.get("r", "A1")
                kind = cell.attrib.get("t")
                value_node = cell.find("m:v", NS)
                inline = cell.find("m:is/m:t", NS)
                value = ""
                if kind == "inlineStr" and inline is not None:
                    value = inline.text or ""
                elif value_node is not None:
                    value = value_node.text or ""
                    if kind == "s":
                        value = shared[int(value)]
                cells[column_index(ref)] = value
            if cells:
                width = max(cells) + 1
                rows.append([cells.get(index, "") for index in range(width)])
        return rows


def check_run_audit() -> dict[str, Any]:
    workbook = DATA_DIR / "tinker_runs_audit_2026-07-12.xlsx"
    runs = read_xlsx_sheet(workbook, "runs")
    key_runs = read_xlsx_sheet(workbook, "key_runs")
    insights = read_xlsx_sheet(workbook, "insights")
    run_count = len(runs) - 1
    gold_count = len(key_runs) - 1
    reconciliation = next((row for row in insights[1:] if row and row[0].startswith("Reconciliation: 983")), None)
    if run_count != 983 or gold_count != 19 or reconciliation is None:
        raise CheckError(f"run-audit reconciliation failed: runs={run_count}, gold={gold_count}")
    detail = reconciliation[1]
    required = ("70+ logged runs", "983 training-run objects", "EVERY create_lora_training_client call")
    if not all(phrase in detail for phrase in required):
        raise CheckError("run-audit reconciliation wording drifted")
    return {
        "status": "PASS",
        "tinker_run_objects": run_count,
        "claim_critical_gold_rows": gold_count,
        "curated_cross_library_claim": "70+ logged runs",
        "reconciliation": detail,
        "plain_english": "983 is the broad infrastructure object count; 70+ is the curated cross-library telemetry corpus. They have different inclusion rules and are not competing totals.",
    }


def render_dashboard(report: dict[str, Any]) -> Path:
    template = (PACKAGE_DIR / "dashboard_template.html").read_text(encoding="utf-8")
    payload = json.dumps(report, ensure_ascii=False).replace("</", "<\\/")
    output = template.replace("__REPORT_JSON__", payload)
    path = OUTPUT_DIR / "dashboard.html"
    path.write_text(output, encoding="utf-8")
    return path


def run_all() -> dict[str, Any]:
    report = {
        "status": "PASS",
        "manifest": check_manifest(),
        "synthetic_fixture": check_synthetic_fixture(),
        "claim2": check_claim2(),
        "p4": check_p4(),
        "run_audit": check_run_audit(),
        "scope": "Offline recomputation of copied project artifacts; no network, model inference, or new training is performed.",
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    render_dashboard(report)
    return report


def print_summary(report: dict[str, Any]) -> None:
    print("OFFLINE DEFENSE CHECKS: PASS")
    print(f"  Provenance: {report['manifest']['file_count']} copied artifacts match SHA-256")
    print("  Claim 2: four runs, 2,560 rollouts per arm")
    for row in report["claim2"]["rows"]:
        print(f"    {row['run']}: last-10 reward={row['late_reward_mean']:.4f}, ZVF={row['late_zvf_mean']:.3f}")
    lo, hi = report["p4"]["exact_contraction_range_pct"]
    print(f"  P4: all six corrected arms contract in length; exact range={lo:.4f}-{hi:.4f}% (say 3.8-12.2%)")
    audit = report["run_audit"]
    print(f"  Run audit: {audit['tinker_run_objects']} objects vs {audit['curated_cross_library_claim']}; {audit['claim_critical_gold_rows']} gold rows")
    print("  Dashboard: submission/demo/defense_fallback/output/dashboard.html")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print the complete report as JSON")
    parser.add_argument("--serve", action="store_true", help="serve the package on localhost after checking")
    parser.add_argument("--port", type=int, default=8771, help="localhost port for --serve")
    args = parser.parse_args(argv)
    try:
        report = run_all()
    except CheckError as exc:
        print(f"OFFLINE DEFENSE CHECKS: FAIL: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_summary(report)
    if args.serve:
        handler = lambda *a, **kw: SimpleHTTPRequestHandler(*a, directory=str(PACKAGE_DIR), **kw)
        server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
        print(f"Open http://127.0.0.1:{args.port}/output/dashboard.html")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
