"""Diagnostic only (not part of the score): does each generated patch `git apply --check` at base commit?"""
import concurrent.futures as cf
import json
from pathlib import Path

import modal

RAW = Path(__file__).resolve().parents[1] / "raw"
rows = {json.loads(l)["instance_id"]: json.loads(l) for l in (RAW / "dataset_subset.jsonl").read_text().splitlines()}
imgs = {x["instance_id"]: x["immutable_uri"] for x in json.loads((RAW / "image_manifest_subset.json").read_text())["images"]}
cands = [c for c in json.loads((RAW / "candidates.json").read_text()) if c["patch"]]
app = modal.App.lookup("pavlov-e1-small-apply-check", create_if_missing=True)


def check(c):
    iid = c["instance_id"]
    sb = modal.Sandbox.create(image=modal.Image.from_registry(imgs[iid]), app=app, timeout=900, block_network=True)
    try:
        sb.filesystem.write_text(c["patch"], "/tmp/p.diff")
        p = sb.exec("bash", "-lc", f"cd /app && git reset -q --hard {rows[iid]['base_commit']} && git apply --check -v /tmp/p.diff")
        p.wait()
        return iid, {"applies": p.returncode == 0, "log": (p.stdout.read() + p.stderr.read())[-1500:]}
    finally:
        sb.terminate()


with cf.ThreadPoolExecutor(5) as ex:
    res = dict(ex.map(check, cands))
(RAW / "apply_check.json").write_text(json.dumps(res, indent=1))
for k, v in res.items():
    print(v["applies"], k)
