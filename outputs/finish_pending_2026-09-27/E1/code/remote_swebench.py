"""Drop-in for `<setup>/.venv/bin/swebench eval ...` that runs the identical native CLI inside a Modal
dockerd sandbox (the 2026-09-12 runtime shape) and brings reports/ + logs/ back next to --report-dir.

Accepts: eval <dataset.jsonl> -p <preds.jsonl> --run-id R --split test -j N --timeout T --report-dir D
Instance images are pre-pulled at the dataset's pinned digest; if that exact digest is no longer
served by Docker Hub, the current tag is pulled and the dataset copy sent to the sandbox is rewritten
to that digest (recorded in image_resolution.json; a deviation, never silent).
"""
import json
import sys
import tarfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import e1_runtime as R  # noqa: E402


def main(argv):
    assert argv[0] == "eval", argv
    dataset = Path(argv[1])
    opt = {argv[i]: argv[i + 1] for i in range(2, len(argv) - 1, 2)}
    report_dir = Path(opt["--report-dir"])
    out_dir = report_dir.parent
    tag = opt["--run-id"]
    rows = [json.loads(l) for l in dataset.read_text().splitlines() if l.strip()]
    sb = R.open_sandbox(timeout=int(opt.get("--sandbox-timeout", 10800)))
    resolution = {"sandbox": sb.object_id, "images": {}}
    try:
        for row in rows:
            want = row["image"]
            rc, _, err = R.sh(sb, f"docker pull -q {want}", 1800)
            if rc == 0:
                resolution["images"][row["instance_id"]] = {"pinned": want, "used": want, "exact": True}
                continue
            tag_ref = want.split("@")[0] + ":latest"
            digest, _ = R.pull(sb, tag_ref)
            R.sh(sb, f"docker tag {tag_ref} {want.split('@')[0]}:latest", 60)
            resolution["images"][row["instance_id"]] = {"pinned": want, "used": digest, "exact": False,
                                                        "pinned_pull_error": err[-300:]}
            row["image"] = digest
        (out_dir / "image_resolution.json").write_text(json.dumps(resolution, indent=2))
        data = "".join(json.dumps(r, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
                       for r in rows).encode()
        receipt = R.run_native(sb, tag, data, Path(opt["-p"]).read_bytes(), out_dir,
                               workers=int(opt["-j"]), timeout=int(opt["--timeout"]))
        with tarfile.open(receipt["receipts_tgz"]) as t:
            t.extractall(out_dir, filter="data")
        return 0 if receipt["returncode"] == 0 else receipt["returncode"]
    finally:
        sb.terminate()
        resolution["terminated_at"] = time.time()
        (out_dir / "image_resolution.json").write_text(json.dumps(resolution, indent=2))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
