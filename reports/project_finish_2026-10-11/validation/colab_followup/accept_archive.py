"""Check the final archive from an isolated extraction, without remote execution."""

from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[4]
REL = Path("reports/project_finish_2026-10-11")
PACK = ROOT / REL
checks = []


def run(name, args, cwd, expected=0):
    result = subprocess.run(args, cwd=cwd, text=True, capture_output=True, timeout=90)
    checks.append(dict(check=name, exit_code=result.returncode,
                       expected_exit_code=expected, stdout=result.stdout,
                       stderr=result.stderr))
    if result.returncode != expected:
        raise RuntimeError(f"{name}: {result.stdout}\n{result.stderr}")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class AuditError(ValueError):
    pass


def require(condition, message):
    if not condition: raise AuditError(message)


def compare_publication(replay, shipped, path="$", tolerance=1e-12):
    """v2 gate: compare the entire publication object, not selected means.

    Object keys, strings, booleans, counts, identities and ordered sequences are
    exact. Only finite floating point values allow absolute tolerance 1e-12;
    no relative tolerance. No ignored publication fields or missing-key default.
    """
    if isinstance(replay, dict):
        require(isinstance(shipped, dict) and replay.keys() == shipped.keys(), path + ": key mismatch")
        for key in replay: compare_publication(replay[key], shipped[key], path + "." + key, tolerance)
    elif isinstance(replay, list):
        require(isinstance(shipped, list) and len(replay) == len(shipped), path + ": sequence mismatch")
        for i, (a, b) in enumerate(zip(replay, shipped)): compare_publication(a, b, f"{path}[{i}]", tolerance)
    elif type(replay) is float:
        require(type(shipped) is float and math.isfinite(replay) and math.isfinite(shipped)
                and abs(replay - shipped) <= tolerance, path + ": finite numeric mismatch")
    else:
        require(type(replay) is type(shipped) and replay == shipped, path + ": exact value mismatch")


def accept_publication(replay, shipped):
    require(replay.get("schema_version") == "samestack-publication-v3-candidate", "unsupported publication schema")
    required = {"schema_version", "grader_version", "protocol", "protocol_sha256", "arms", "contrasts", "item_level", "power", "selected_inputs", "holm_family", "family_status", "confirmatory_publication_allowed", "claim_evidence", "actual_design_power"}
    require(required <= replay.keys(), "incomplete publication schema")
    family = {"g8-ppo", "g8-ppo_8x8", "g8-g2", "g2-ppo"}
    for source in ("stored", "mine"):
        require(source in replay["contrasts"] and set(replay["contrasts"][source]) == family, "incomplete contrast inventory")
        for contrast in replay["contrasts"][source].values():
            require({"n", "seeds", "diffs", "mean", "sd", "t_ci", "boot_ci", "p_t", "p_signflip", "tost_p"} <= contrast.keys(), "incomplete inferential schema")
    require(set(replay["power"]) == set(replay["item_level"]) == family, "incomplete power/item inventory")
    compare_publication(replay, shipped)
    require(replay.get("family_status") == "COMPLETE_PRIMARY_FAMILY", "INCOMPLETE_PRIMARY_FAMILY")
    require(replay.get("confirmatory_publication_allowed") is True, "unbound publication cohort")
    require(replay.get("holm_family") == ["g8-ppo", "g8-ppo_8x8", "g8-g2"], "frozen family mismatch")
    for phase in ("pre", "post"):
        count = sum(v["item_disagreements_" + phase + "_total"] for v in replay["arms"].values())
        require(count == 0, phase + " independent regrading disagreement")
    # A calibration object, when shipped, is included in the recursive comparison.
    # Absence does not establish calibration or authorize a transport claim.
    return {"schema_version": "publication-replay-gate-v2", "status": "PASS",
            "numeric_absolute_tolerance": 1e-12,
            "platform_calibration_verified": False,
            "platform_fields_compared": "platform_calibrated" in replay,
            "platform_scope": "numeric replay only; calibration schema and transport validity are outside this same-stack gate"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=ROOT / "dist/TinkerRL_PES_Review_2026-10-11.zip")
    parser.add_argument('--receipt', type=Path, required=True, help='NEW candidate receipt; existing receipts are never overwritten')
    args = parser.parse_args(argv)
    require(not args.receipt.exists(), 'refuse to overwrite historical acceptance receipt')
    archive = args.archive
    run("repository manifest", [sys.executable, str(REL / "verify_review_pack.py")], ROOT)
    with tempfile.TemporaryDirectory(prefix="pes-colab-final-") as directory:
        extracted = Path(directory)
        with zipfile.ZipFile(archive) as z:
            require(z.testzip() is None, "ZIP CRC mismatch")
            require(all(not Path(n).is_absolute() and ".." not in Path(n).parts
                       for n in z.namelist()), "unsafe archive path")
            checks.append(dict(check="ZIP CRC and safe paths", status="PASS", entries=len(z.namelist())))
            z.extractall(extracted)
        run("extracted manifest", [sys.executable, str(REL / "verify_review_pack.py")], extracted)
        run("extracted demo self-tests", [sys.executable, "submission/demo/run_demo.py", "--self-test"], extracted)
        run("extracted offline demo", [sys.executable, "submission/demo/run_demo.py"], extracted)
        run("extracted registry comparison", [sys.executable, "platform_hybrid/registry/query.py", "stackdiff",
            "colab-open_dapo_e3", "tinker_dapo_qwen3.5-4b_gsm8k", "--exit-level"], extracted, expected=5)
        v2 = extracted / REL / "validation/samestack_v2"
        result = subprocess.run([sys.executable, "-I", str(v2 / "legacy_v2_analysis.py"), str(v2),
                                 "platform_hybrid/experiments/results/samestack_gsm8k_v2s",
                                 str(v2 / "gsm8k_test_gold.json"), "--protocol", str(v2 / "publication_protocol_v3.json")],
                                cwd=extracted, text=True, capture_output=True, timeout=600)
        require(result.returncode == 0, result.stderr)
        replay, shipped = json.loads(result.stdout), json.loads((v2 / "v2_analysis.json").read_text())
        gate = accept_publication(replay, shipped)
        checks.append(dict(check="extracted same-stack full publication replay", **gate))
        audit = extracted / REL / "colab_validation"
        output = extracted / "replayed_colab_result.json"
        run("extracted frozen source replay", [sys.executable, str(audit / "analyze.py"),
            "--input-dir", str(audit / "inputs"), "--output", str(output)], extracted)
        run("extracted analysis acceptance tests", [sys.executable, "-m", "unittest", "discover",
            "-s", str(audit), "-p", "test_analyze.py"], extracted)
        spec = importlib.util.spec_from_file_location("colab_replay_check", audit / "run_colab.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.compare(json.loads(output.read_text()), json.loads((audit / "result.json").read_text()))
        checks.append(dict(check="replayed result versus delivered Colab result", status="PASS", numeric_tolerance=1e-12))
        thesis = extracted / REL / "thesis"
        inventory = json.loads((thesis / "public_source_availability.json").read_text())
        for row in inventory["files"]:
            target = thesis / row["path"]
            require(target.stat().st_size == row["bytes"] and digest(target) == row["sha256"], row["path"])
        checks.append(dict(check="thesis source inventory", status="PASS", files=len(inventory["files"])))
        readme = extracted / REL / "README.md"
        readme.write_text(readme.read_text() + "\nIntentional local integrity probe.\n")
        run("deliberate tamper rejected", [sys.executable, str(REL / "verify_review_pack.py")], extracted, expected=1)
    run("frozen publication paths unchanged", ["git", "diff", "--exit-code", "--", "README.md",
        "reports/public_revision_2026-10-03", "reports/public_revision_2026-10-04", "PUBLICATION_MANIFEST.json"], ROOT)
    receipt = dict(schema="pes-review-archive-acceptance-v2-candidate", status="PASS", archive=archive.name,
                   sha256=digest(archive), bytes=archive.stat().st_size, checks=checks,
                   scope="Exact archive extraction, manifest and source-inventory verification, offline demo, registry example (revised CLI, --exit-level), same-stack v2 replay and re-grade, frozen Colab analysis replay and tamper rejection. No remote execution or original scientific authenticity claim.")
    with args.receipt.open("x") as stream:
        stream.write(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k:receipt[k] for k in ("status", "archive", "sha256", "bytes")}, indent=2))
    print(f"{len(checks)} archive checks passed")


if __name__ == "__main__":
    main()
