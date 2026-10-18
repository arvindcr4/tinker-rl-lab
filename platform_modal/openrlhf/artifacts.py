"""Descriptor-pinned regular-file snapshots for native run artifacts."""

import os
import stat
from pathlib import Path


def read_regular_bytes(path: Path) -> bytes:
    """Read one regular file without following its final component.

    Parent directories are trusted by the caller; O_NOFOLLOW does not prevent
    traversing symlinked parents. This is not a path-confinement primitive.
    """
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_NONBLOCK"):
        raise RuntimeError("native artifact verification requires POSIX no-follow/nonblocking open")
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as exc:
        raise RuntimeError("native artifact must be a readable regular file") from exc
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise RuntimeError("native artifact must be a regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            return stream.read()
    finally:
        os.close(descriptor)
