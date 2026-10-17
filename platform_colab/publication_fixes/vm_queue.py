"""Bounded queue for trusted shell-command job specifications.

Shell commands are intentionally executable trusted input. Validate all identities
and destinations before spawning anything; never reuse a bare done-file as proof.
"""

import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path


def validate_jobs(jobs, concurrency):
    if type(concurrency) is not int or concurrency < 1:
        raise ValueError("Concurrency must be positive")
    if not isinstance(jobs, list):
        raise ValueError("Jobs must be a list")
    names, outputs = set(), set()
    for job in jobs:
        name = job.get("name", "")
        if (
            not isinstance(name, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name)
            or name in names
        ):
            raise ValueError("Job names must be unique safe basenames")
        names.add(name)
        if not isinstance(job.get("cmd"), str) or not job["cmd"].strip():
            raise ValueError("Each job needs a trusted nonempty shell command")
        timeout = job.get("timeout_s", 10800)
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Job timeout must be finite and positive")
        for key in ("done", "output"):
            if job.get(key):
                destination = str(Path(job[key]).resolve())
                if destination in outputs:
                    raise ValueError("Job output/done paths must not collide")
                outputs.add(destination)
    return jobs


def run_queue(jobs, concurrency, logdir, poll_s=1):
    validate_jobs(jobs, concurrency)
    logdir = Path(logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    status = {j["name"]: {"state": "queued"} for j in jobs}
    pending, running = list(jobs), {}
    interrupted = None
    previous_handlers = {}

    def request_stop(signum, frame):
        # Do not raise here: a signal can arrive between Popen and ownership
        # registration. Finish registration, then leave the loop for cleanup.
        nonlocal interrupted
        interrupted = interrupted or signum

    def stop_process_group(proc):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()

    def dump():
        path = logdir / "status.json"
        path.with_suffix(".tmp").write_text(json.dumps(status, indent=1))
        path.with_suffix(".tmp").replace(path)

    try:
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            previous_handlers[signum] = signal.signal(signum, request_stop)
        while (pending or running) and interrupted is None:
            while pending and len(running) < concurrency and interrupted is None:
                job = pending.pop(0)
                handle = (logdir / (job["name"] + ".log")).open("w")
                try:
                    proc = subprocess.Popen(
                        job["cmd"],
                        shell=True,
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                        env={
                            **os.environ,
                            "PF_CONCURRENCY": str(concurrency),
                            **job.get("env", {}),
                        },
                    )
                except BaseException:
                    handle.close()
                    raise
                running[job["name"]] = (
                    proc,
                    handle,
                    time.monotonic() + job.get("timeout_s", 10800),
                )
                status[job["name"]] = {"state": "running", "start": time.time()}
                dump()
            for name, (proc, handle, deadline) in list(running.items()):
                rc = proc.poll()
                if rc is None and time.monotonic() >= deadline:
                    stop_process_group(proc)
                    rc = 124
                if rc is not None:
                    handle.close()
                    status[name].update(state="ok" if rc == 0 else "failed", rc=rc, end=time.time())
                    del running[name]
                    dump()
            if pending or running:
                # Bound cancellation latency even for an unusually long poll.
                time.sleep(min(poll_s, 0.1))
    finally:
        try:
            for name, (proc, handle, _) in running.items():
                stop_process_group(proc)
                handle.close()
                status[name].update(state="cancelled", rc=128 + (interrupted or 1), end=time.time())
            for job in pending:
                status[job["name"]].update(state="cancelled")
            if interrupted is not None:
                status["_interrupted_signal"] = interrupted
            dump()
        finally:
            for signum, previous in previous_handlers.items():
                signal.signal(signum, previous)
    status["_all_done"] = time.time()
    dump()
    if interrupted is not None:
        return 128 + interrupted
    return int(
        any(value.get("state") == "failed" for value in status.values() if isinstance(value, dict))
    )


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    path = Path(argv[0])
    return run_queue(json.loads(path.read_text()), int(argv[1]), path.parent / "logs")


if __name__ == "__main__":
    raise SystemExit(main())
