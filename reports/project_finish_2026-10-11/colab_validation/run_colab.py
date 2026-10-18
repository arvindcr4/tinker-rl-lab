#!/usr/bin/env python3
"""Replay the frozen public numeric analysis in an ephemeral Colab CPU VM."""

from pathlib import Path
import base64
import datetime
import hashlib
import io
import json
import math
import subprocess
import tempfile
import zipfile

BASE = Path(__file__).resolve().parent
STAGE = BASE.parent / "validation/colab_followup"
STAGE.mkdir(parents=True, exist_ok=True)
SESSION = "pes-zvf-lineage-20261011"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(a, b, at="result"):
    if isinstance(a, dict):
        if not isinstance(b, dict) or a.keys() != b.keys():
            raise ValueError("Different keys at " + at)
        for key in a:
            compare(a[key], b[key], at + "." + key)
    elif isinstance(a, list):
        if not isinstance(b, list) or len(a) != len(b):
            raise ValueError("Different list at " + at)
        for i, (x, y) in enumerate(zip(a, b, strict=False)):
            compare(x, y, at + f"[{i}]")
    elif isinstance(a, float):
        if not isinstance(b, (float, int)) or not math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("Different numeric value at " + at)
    elif a != b:
        raise ValueError("Different value at " + at)


def main():
    # Only explicitly selected numeric inputs, source excerpts and analyzer travel.
    paths = [BASE / "analyze.py", BASE / "PROTOCOL.md", *sorted((BASE / "inputs").glob("*"))]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in paths:
            if not f.is_file():
                raise ValueError("Unexpected bundle entry")
            z.writestr(f.relative_to(BASE).as_posix(), f.read_bytes())
    payload = buf.getvalue()
    payload_sha = hashlib.sha256(payload).hexdigest()
    code = """import base64,hashlib,io,json,os,platform,runpy,sys,time,zipfile
from pathlib import Path
import numpy,scipy
started=time.time()
payload=base64.b64decode(PAYLOAD)
assert hashlib.sha256(payload).hexdigest()==PAYLOAD_SHA
root=Path('/content/pes_zvf_lineage_20261011')
root.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(io.BytesIO(payload)) as z:
    assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist())
    z.extractall(root)
sys.argv=[str(root/'analyze.py'),'--input-dir',str(root/'inputs'),'--output',str(root/'result.json')]
runpy.run_path(str(root/'analyze.py'),run_name='__main__')
result=json.loads((root/'result.json').read_text())
receipt={'result':result,'runtime':{'python':platform.python_version(),'platform':platform.platform(),'numpy':numpy.__version__,'scipy':scipy.__version__,'cpu_count':os.cpu_count()},'payload_sha256':PAYLOAD_SHA,'remote_elapsed_seconds':time.time()-started,'result_file_sha256':hashlib.sha256((root/'result.json').read_bytes()).hexdigest()}
print('COLAB_RESULT_JSON='+json.dumps(receipt,separators=(',',':'),allow_nan=False))
""".replace("PAYLOAD_SHA", repr(payload_sha)).replace(
        "PAYLOAD", repr(base64.b64encode(payload).decode())
    )
    wrapper = STAGE / "remote_job.py"
    wrapper.write_text(code)
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with tempfile.TemporaryDirectory(prefix="pes-colab-state-") as temp:
        config = str(Path(temp) / "sessions.json")
        cmd = ["colab", "--config", config, "run", "-s", SESSION, "--timeout", "180", str(wrapper)]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=420)
        except subprocess.TimeoutExpired:
            subprocess.run(
                ["colab", "--config", config, "stop", "-s", SESSION],
                capture_output=True,
                text=True,
                timeout=60,
            )
            raise
        (STAGE / "colab_stdout.log").write_text(result.stdout)
        (STAGE / "colab_stderr.log").write_text(result.stderr)
        if result.returncode:
            subprocess.run(
                ["colab", "--config", config, "stop", "-s", SESSION],
                capture_output=True,
                text=True,
                timeout=60,
            )
            raise RuntimeError("Colab failed; inspect local stdout/stderr logs")
        records = [
            line.removeprefix("COLAB_RESULT_JSON=")
            for line in result.stdout.splitlines()
            if line.startswith("COLAB_RESULT_JSON=")
        ]
        if len(records) != 1:
            raise ValueError("Missing or duplicate remote result")
        envelope = json.loads(records[0])
        remote = envelope.pop("result")
        compare(json.loads((STAGE / "local_result.json").read_text()), remote)
        (BASE / "result.json").write_text(
            json.dumps(remote, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        )
        state = subprocess.run(
            ["colab", "--config", config, "sessions"], capture_output=True, text=True, timeout=45
        )
        (STAGE / "cleanup.log").write_text(state.stdout + state.stderr)
        if (
            state.returncode
            or "No active sessions found on server" not in state.stdout + state.stderr
        ):
            raise ValueError("Automatic cleanup not confirmed; inspect cleanup.log")
        envelope.update(
            status="PASS",
            session=SESSION,
            resource="CPU; no accelerator requested",
            started_at_utc=started,
            finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            execution_timeout_seconds=180,
            outer_timeout_seconds=420,
            colab_exit_code=result.returncode,
            analyzer_sha256=digest(BASE / "analyze.py"),
            protocol_sha256=digest(BASE / "PROTOCOL.md"),
            manifest_sha256=digest(BASE / "inputs/manifest.json"),
            local_remote_comparison="PASS; exact structure/identities, numeric tolerance 1e-12",
            cleanup="PASS; CLI confirms no active sessions",
            local_result_sha256=digest(STAGE / "local_result.json"),
            delivered_result_sha256=digest(BASE / "result.json"),
            command="colab --config <isolated temporary state> run -s "
            + SESSION
            + " --timeout 180 <frozen bundle wrapper>",
        )
        (BASE / "execution_receipt.json").write_text(json.dumps(envelope, indent=2) + "\n")
        print(
            json.dumps(
                {
                    k: envelope[k]
                    for k in ["status", "resource", "runtime", "remote_elapsed_seconds", "cleanup"]
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
