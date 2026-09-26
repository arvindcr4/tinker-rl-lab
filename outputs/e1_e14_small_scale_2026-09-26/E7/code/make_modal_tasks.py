"""Copy selected BinaryAudit tasks (upstream cbd86c7, unmodified instruction/tests/task.toml)
into code/tasks_modal/, inlining docker/base.Dockerfile as a named build stage so the
environment builds on Modal (which cannot see the locally-tagged binaryaudit-base image).
Only change: `FROM binaryaudit-base:latest` -> `FROM binaryaudit-base` (stage reference)."""
import json, shutil, pathlib
root = pathlib.Path(__file__).parent
src = root / "BinaryAudit"
base = (src / "docker/base.Dockerfile").read_text().replace(
    "FROM --platform=linux/amd64 ubuntu:24.04", "FROM --platform=linux/amd64 ubuntu:24.04 AS binaryaudit-base", 1)
sel = json.load(open(root.parent / "raw/selection.json"))["selected"]
import os, sys
out = root / os.environ.get("TASKS_OUT", "tasks_modal")
if len(sys.argv) > 1:
    sel = [t for t in sel if t in sys.argv[1:]]
shutil.rmtree(out, ignore_errors=True)
for t in sel:
    shutil.copytree(src / "tasks" / t, out / t)
    df = out / t / "environment/Dockerfile"
    txt = df.read_text()
    assert txt.count("FROM binaryaudit-base:latest") == 1, t
    txt = txt.replace("FROM binaryaudit-base:latest", "FROM binaryaudit-base")
    # Environment-rot fix (2026-09-26): the debian:11 security pool on deb.debian.org is purged while its index
    # still lists deb11uN packages -> apt 404s. Point the builder stage at the image's own pinned snapshot
    # (lines ship commented in /etc/apt/sources.list), keeping the same clang/toolchain as upstream intended.
    txt = txt.replace("debian:11-slim AS builder\n", "debian:11-slim AS builder\n"
        "RUN sed -i -e 's|^# deb http://snapshot|deb [check-valid-until=no] http://snapshot|' "
        "-e '/^deb http:\\/\\/deb.debian.org/d' /etc/apt/sources.list\n", 1)
    df.write_text(base + "\n" + txt)
print(sorted(p.name for p in out.iterdir()))
