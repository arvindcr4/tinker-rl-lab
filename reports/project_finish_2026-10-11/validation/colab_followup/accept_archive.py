"""Check the final archive from an isolated extraction, without remote execution."""

from pathlib import Path
import hashlib
import importlib.util
import json
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


def main():
    archive = ROOT / "dist/TinkerRL_PES_Review_2026-10-11.zip"
    run("repository manifest", [sys.executable, str(REL / "verify_review_pack.py")], ROOT)
    with tempfile.TemporaryDirectory(prefix="pes-colab-final-") as directory:
        extracted = Path(directory)
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None
            assert all(not Path(n).is_absolute() and ".." not in Path(n).parts
                       for n in z.namelist())
            checks.append(dict(check="ZIP CRC and safe paths", status="PASS", entries=len(z.namelist())))
            z.extractall(extracted)
        run("extracted manifest", [sys.executable, str(REL / "verify_review_pack.py")], extracted)
        run("extracted demo self-tests", [sys.executable, "submission/demo/run_demo.py", "--self-test"], extracted)
        run("extracted offline demo", [sys.executable, "submission/demo/run_demo.py"], extracted)
        run("extracted registry comparison", [sys.executable, "platform_hybrid/registry/query.py", "stackdiff",
            "colab-open_dapo_e3", "tinker_dapo_qwen3.5-4b_gsm8k", "--exit-level"], extracted, expected=5)
        v2 = extracted / REL / "validation/samestack_v2"
        result = subprocess.run([sys.executable, "-I", str(v2 / "v2_analysis.py"), str(v2),
                                 "platform_hybrid/experiments/results/samestack_gsm8k_v2s",
                                 str(v2 / "gsm8k_test_gold.json")],
                                cwd=extracted, text=True, capture_output=True, timeout=600)
        assert result.returncode == 0, result.stderr
        replay, shipped = json.loads(result.stdout), json.loads((v2 / "v2_analysis.json").read_text())
        for k in ("g8-ppo", "g8-g2", "g2-ppo"):
            a, b = replay["contrasts"]["stored"][k], shipped["contrasts"]["stored"][k]
            assert abs(a["mean"] - b["mean"]) < 1e-12 and a["n"] == b["n"], k
        disagreements = sum(v["item_disagreements_post_total"] for v in replay["arms"].values())
        assert disagreements == 0, disagreements
        checks.append(dict(check="extracted same-stack v2 replay", status="PASS",
                           contrasts={k: replay["contrasts"]["stored"][k]["mean"] for k in ("g8-ppo", "g8-g2", "g2-ppo")},
                           regrade_disagreements=disagreements))
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
            assert target.stat().st_size == row["bytes"] and digest(target) == row["sha256"], row["path"]
        checks.append(dict(check="thesis source inventory", status="PASS", files=len(inventory["files"])))
        readme = extracted / REL / "README.md"
        readme.write_text(readme.read_text() + "\nIntentional local integrity probe.\n")
        run("deliberate tamper rejected", [sys.executable, str(REL / "verify_review_pack.py")], extracted, expected=1)
    run("frozen publication paths unchanged", ["git", "diff", "--exit-code", "--", "README.md",
        "reports/public_revision_2026-10-03", "reports/public_revision_2026-10-04", "PUBLICATION_MANIFEST.json"], ROOT)
    receipt = dict(schema="pes-review-archive-acceptance-v1", status="PASS", archive=archive.name,
                   sha256=digest(archive), bytes=archive.stat().st_size, checks=checks,
                   scope="Exact archive extraction, manifest and source-inventory verification, offline demo, registry example (revised CLI, --exit-level), same-stack v2 replay and re-grade, frozen Colab analysis replay and tamper rejection. No remote execution or original scientific authenticity claim.")
    (PACK / "validation/archive_checks.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k:receipt[k] for k in ("status", "archive", "sha256", "bytes")}, indent=2))
    print(f"{len(checks)} archive checks passed")


if __name__ == "__main__":
    main()
