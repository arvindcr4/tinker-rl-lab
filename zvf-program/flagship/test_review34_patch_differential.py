"""Real local git differential fixtures; no native SWE grader or model execution."""
import subprocess

import pytest

from modal_e1_swe_bench_pro_full import _recount_hunks, _validate_unified_diff


def apply(directory, files, patch, *, recount):
    directory.mkdir()
    subprocess.run(["git", "init", "-q", str(directory)], check=True)
    for name, content in files.items():
        (directory / name).write_bytes(content)
    command = ["git", "apply"] + (["--recount"] if recount else []) + ["-"]
    result = subprocess.run(command, cwd=directory, input=patch.encode(), capture_output=True)
    return result.returncode, {name: (directory / name).read_bytes() for name in files}


def diff(name, header, body):
    return f"diff --git a/{name} b/{name}\n--- a/{name}\n+++ b/{name}\n{header}\n{body}"


@pytest.mark.parametrize("patch,files,expected", [
    (diff("a.txt", "@@ -1,99 +1,88 @@", " one\n-two\n+TWO\n three\n"),
     {"a.txt": b"one\ntwo\nthree\n"}, {"a.txt": b"one\nTWO\nthree\n"}),
    (diff("a.txt", "@@ -1,99 +1,88 @@", "-one\n+ONE\n")
     + diff("b.txt", "@@ -1,9 +1,9 @@", "-two\n+TWO\n"),
     {"a.txt": b"one\n", "b.txt": b"two\n"}, {"a.txt": b"ONE\n", "b.txt": b"TWO\n"}),
    (diff("a.txt", "@@ -1,9 +1,9 @@", " one\n+two\n"),
     {"a.txt": b"one\n"}, {"a.txt": b"one\ntwo\n"}),
    (diff("a.txt", "@@ -1,9 +1,9 @@", "-one\n two\n"),
     {"a.txt": b"one\ntwo\n"}, {"a.txt": b"two\n"}),
    (diff("a.txt", "@@ -1,9 +1,9 @@", "-one\n\\ No newline at end of file\n+ONE\n\\ No newline at end of file\n"),
     {"a.txt": b"one"}, {"a.txt": b"ONE"}),
])
def test_count_only_repairs_match_git_recount_bytes(tmp_path, patch, files, expected):
    repaired, changes = _recount_hunks(patch)
    assert changes > 0
    raw_status, raw_files = apply(tmp_path / "native", files, patch, recount=True)
    repaired_status, repaired_files = apply(tmp_path / "adapter", files, repaired, recount=False)
    assert (raw_status, repaired_status) == (0, 0)
    assert raw_files == repaired_files == expected


@pytest.mark.parametrize("patch", ["", diff("a.txt", "@@ malformed @@", "-one\n+ONE\n"),
    diff("a.txt", "@@ -1,9 +1,9 @@", "-absent\n+ONE\n"),
    diff("a.txt", "@@ -1,9 +1,9 @@", "-one\n+ONE\n?illegal body\n")])
def test_empty_malformed_and_nonapplicable_are_not_success(tmp_path, patch):
    files = {"a.txt": b"one\n"}
    repaired, _ = _recount_hunks(patch)
    native, native_files = apply(tmp_path / "native", files, patch, recount=True)
    adapter, adapter_files = apply(tmp_path / "adapter", files, repaired, recount=False)
    assert native != 0
    # Plain git can ignore trailing garbage after repaired counts. The
    # production validation gate must reject it before submission.
    assert adapter != 0 or not _validate_unified_diff(repaired)[0]
    if adapter != 0:
        assert native_files == adapter_files == files


def test_context_only_hunks_are_explicit_adapter_policy_not_git_equivalence(tmp_path):
    patch = diff("a.txt", "@@ -1,1 +1,1 @@", " one\n")
    repaired, changes = _recount_hunks(patch)
    assert repaired == "" and changes > 0
    native, native_files = apply(tmp_path / "native", {"a.txt": b"one\n"}, patch, recount=True)
    adapter, adapter_files = apply(tmp_path / "adapter", {"a.txt": b"one\n"}, repaired, recount=False)
    # git --recount accepts context-only input; the adapter deliberately drops
    # it and its validator rejects an empty submission. This is not parity.
    assert native == 0 and adapter != 0
    assert not _validate_unified_diff(repaired)[0]
    assert native_files == adapter_files == {"a.txt": b"one\n"}
