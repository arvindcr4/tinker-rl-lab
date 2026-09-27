"""Recover native swebench reports from sandboxes orphaned when the previous E1 driver process was killed
(~01:54 UTC). The native eval kept running server-side and completed; stdout/stderr of that exec were lost
with the client, so only reports/ + logs/ are recovered. Then terminate every sandbox of the lane app."""
import json
import sys
import tarfile
import time
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).parent))
import e1_runtime as R  # noqa: E402

OUT = Path(__file__).parent.parent / "remaining"
app = modal.App.lookup(R.APP_NAME)
for sb in modal.Sandbox.list(app_id=app.app_id):
    rc, out, _ = R.sh(sb, "ls /e1/reports 2>/dev/null", 60)
    reps = [l for l in out.split() if l.endswith(".json")]
    if reps:
        tag = reps[0].rsplit(".", 2)[-2]
        bdir = OUT / tag
        assert not list((bdir / "reports").glob("*.json")) if (bdir / "reports").exists() else True
        R.sh(sb, f"cd /e1 && tar czf /tmp/{tag}_receipts.tgz reports logs", 300)
        tgz = bdir / f"{tag}_native_receipts.tgz"
        sb.filesystem.copy_to_local(f"/tmp/{tag}_receipts.tgz", str(tgz))
        with tarfile.open(tgz) as t:
            t.extractall(bdir, filter="data")
        (bdir / f"{tag}_native_exit.json").write_text(json.dumps({
            "returncode": None, "sandbox": sb.object_id, "harvested_at": time.time(),
            "note": "native eval completed in sandbox orphaned by killed driver; report harvested; "
                    "exec stdout/stderr/returncode lost"}, indent=2))
        st = json.load(open(bdir / f"{tag}_status.json"))
        st["finished_at"] = time.time()
        st["harvested_from_orphan"] = True
        (bdir / f"{tag}_status.json").write_text(json.dumps(st, indent=2))
        print("HARVESTED", tag, sb.object_id)
    sb.terminate()
    print("TERMINATED", sb.object_id)
