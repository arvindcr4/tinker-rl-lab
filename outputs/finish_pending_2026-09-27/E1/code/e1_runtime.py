"""E1 finish-0927 native runtime: Modal CPU sandbox (vm_runtime) running dockerd on the cached
native image im-pAVtHHKH6EXHe7tlbHI7kb (/native/.venv swebench 5.0.2), same shape as the
2026-09-12 runtime0X sandboxes (Sandbox.create('/usr/sbin/dockerd', cpu=(4,4), memory=16 GiB,
experimental_options={'vm_runtime': True})).

Also re-implements the base-commit source-context collector used by the 2026-09-12 multilingual
waves (its driver metadata_wave08.py was lost); selection logic is imported verbatim from
zvf-program/flagship/modal_e1_swe_bench_pro_full.py (_search_terms, _path_candidates,
_score_inventory_path, _slice_source, SOURCE_EXTENSIONS, MAX_SOURCE_*), and file contents are read
with `git -C /testbed show <base>:<path>` inside the pinned instance image, as the recorded
container events show.
"""
from __future__ import annotations

import base64
import json
import shlex
import sys
import time
import types
from pathlib import Path

import modal

REPO = Path("/Users/arvind/Developer/agentic_repos/tinker-rl-lab")
APP_NAME = "pavlov-e1-finish-native-0927"
NATIVE_IMAGE_ID = "im-pAVtHHKH6EXHe7tlbHI7kb"


def pro_flagship():
    """Import the Pro flagship module for its pure selection helpers (modal import is real)."""
    sys.path.insert(0, str(REPO / "zvf-program/flagship"))
    import modal_e1_swe_bench_pro_full as f  # noqa: E402
    return f


def open_sandbox(timeout: int = 10800, log=print):
    app = modal.App.lookup(APP_NAME, create_if_missing=True)
    image = modal.Image.from_id(NATIVE_IMAGE_ID)
    sb = modal.Sandbox.create("/usr/sbin/dockerd", app=app, image=image, cpu=(4, 4),
                              memory=(16384, 16384), timeout=timeout,
                              experimental_options={"vm_runtime": True})
    log(f"SANDBOX {sb.object_id}")
    for _ in range(90):
        rc, out, err = sh(sb, "docker info >/dev/null 2>&1 && echo ok", 30)
        if "ok" in out:
            log(f"DOCKER_READY {sb.object_id}")
            return sb
        time.sleep(2)
    sb.terminate()
    raise RuntimeError("dockerd did not become ready")


def sh(sb, cmd: str, timeout: int = 600, text: bool = True):
    p = sb.exec("bash", "-lc", cmd, timeout=timeout, text=text)
    out = p.stdout.read()
    err = p.stderr.read()
    p.wait()
    return p.returncode, out, err


def pull(sb, image: str) -> str:
    """Pull an instance image; return its immutable repo digest reference."""
    for attempt in range(3):
        rc, out, err = sh(sb, f"docker pull -q {shlex.quote(image)}", 1800)
        if rc == 0:
            break
        time.sleep(10 * (attempt + 1))
    else:
        raise RuntimeError(f"pull failed {image}: {err[-500:]}")
    rc, out, err = sh(sb, f"docker image inspect {shlex.quote(image)}", 60)
    inspect = json.loads(out)[0]
    repo = image.split(":")[0].split("@")[0]
    digests = [d for d in inspect.get("RepoDigests") or [] if d.startswith(repo + "@")]
    if not digests:
        raise RuntimeError(f"no repo digest for {image}")
    return digests[0], inspect


def collect_sources(sb, row: dict, f) -> dict:
    """Base-commit source context for one task (Pro flagship selection rules)."""
    iid, base = row["instance_id"], row["base_commit"]
    image = row["image"]
    cname = "src_" + iid.replace(".", "_")
    rc, out, err = sh(sb, f"docker rm -f {cname} >/dev/null 2>&1; docker run -d --name {cname} "
                          f"--entrypoint sleep {shlex.quote(image)} 7200", 300)
    if rc != 0:
        raise RuntimeError(f"container start failed {iid}: {err[-400:]}")
    container = out.strip()
    try:
        def dex(cmd, t=300, text=True):
            return sh(sb, f"docker exec {cname} bash -c {shlex.quote(cmd)}", t, text)
        rc, head, _ = dex("git -C /testbed rev-parse HEAD")
        rc, inv, err = dex("git -C /testbed ls-files -z", 300)
        if rc != 0:
            raise RuntimeError(f"ls-files failed {iid}: {err[-300:]}")
        inventory = [p for p in inv.split("\0") if p and Path(p).suffix.lower() in f.SOURCE_EXTENSIONS]
        inventory_set = set(inventory)
        task_text = str(row.get("problem_statement") or "")
        # Pro joins problem_statement/requirements/interface; multilingual rows have only the first.
        task_text = "\n".join([task_text, "", ""])
        terms = f._search_terms(task_text)
        selected = [p for p in f._path_candidates(task_text) if p in inventory_set]
        if terms:
            args = " ".join(f"-e {shlex.quote(t)}" for t in terms[:8])
            rc, gout, _ = dex(f"git -C /testbed grep -I -l {args} --", 300)
            if rc in (0, 1):
                for p in gout.splitlines():
                    if p in inventory_set and p not in selected:
                        selected.append(p)
        ranked = sorted(inventory, key=lambda p: (-f._score_inventory_path(p, terms), len(p), p))
        for p in ranked:
            if len(selected) >= f.MAX_SOURCE_FILES:
                break
            if f._score_inventory_path(p, terms) <= 0 and selected:
                break
            if p not in selected:
                selected.append(p)
        sources, chars = {}, 0
        for p in selected[: f.MAX_SOURCE_FILES]:
            rc, raw, _ = dex(f"git -C /testbed show {shlex.quote(base + ':' + p)} | base64 -w0", 120)
            if rc != 0:
                continue
            try:
                content = base64.b64decode(raw).decode("utf-8")
            except (UnicodeDecodeError, ValueError):
                continue
            sliced = f._slice_source(content, terms)
            remaining = f.MAX_SOURCE_CHARS - chars
            if remaining <= 0:
                break
            if len(sliced) > remaining:
                sliced = sliced[:remaining]
            if sliced:
                sources[p] = sliced
                chars += len(sliced)
        if not sources:
            raise RuntimeError(f"no base-commit source context for {iid}")
        return {"files": sources, "receipt": {"base_verified": head.strip(), "container": container,
                "search_terms": terms, "file_count": len(sources), "source_chars": chars,
                "collected_at": time.time()}}
    finally:
        sh(sb, f"docker rm -f {cname} >/dev/null 2>&1", 120)


def run_native(sb, tag: str, dataset_jsonl: bytes, predictions_jsonl: bytes, local_out: Path,
               workers: int = 4, timeout: int = 1800, wall: int = 7500, log=print) -> dict:
    """Run the native swebench CLI inside the sandbox; fetch reports + logs to local_out."""
    sh(sb, "mkdir -p /e1/reports", 60)
    sb.filesystem.write_bytes(dataset_jsonl, f"/e1/{tag}_native_dataset.jsonl")
    sb.filesystem.write_bytes(predictions_jsonl, f"/e1/{tag}_predictions.jsonl")
    cmd = ["/native/.venv/bin/swebench", "eval", f"/e1/{tag}_native_dataset.jsonl", "-p",
           f"/e1/{tag}_predictions.jsonl", "--run-id", tag, "--split", "test", "-j", str(workers),
           "--timeout", str(timeout), "--report-dir", "/e1/reports"]
    started = time.time()
    rc, out, err = sh(sb, "cd /e1 && " + " ".join(shlex.quote(c) for c in cmd), wall)
    finished = time.time()
    local_out.mkdir(parents=True, exist_ok=True)
    (local_out / f"{tag}_native_stdout.txt").write_text(out)
    (local_out / f"{tag}_native_stderr.txt").write_text(err)
    rc2, _, e2 = sh(sb, f"cd /e1 && tar czf /tmp/{tag}_receipts.tgz reports logs 2>/dev/null; ls -la /tmp/{tag}_receipts.tgz", 300)
    tgz = local_out / f"{tag}_native_receipts.tgz"
    sb.filesystem.copy_to_local(f"/tmp/{tag}_receipts.tgz", str(tgz))
    receipt = {"command": cmd, "returncode": rc, "started_at": started, "finished_at": finished,
               "sandbox": sb.object_id, "receipts_tgz": str(tgz)}
    (local_out / f"{tag}_native_exit.json").write_text(json.dumps(receipt, indent=2))
    log(f"NATIVE {tag} rc={rc} {int(finished - started)}s")
    return receipt
