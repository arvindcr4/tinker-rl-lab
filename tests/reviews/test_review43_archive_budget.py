"""Aggregate-budget regressions: temporary files and disabled provider fakes only."""
import json
import sys
from types import SimpleNamespace

import pytest

from platform_hybrid.experiments import archive_local_artifacts as archive


def test_oversize_rejected_before_payload_write(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "big").write_bytes(b"x" * 11)
    budget = archive.ArchiveBudget(10)
    result = archive.copy_tree_hardlinked(source, tmp_path / "out", budget)
    assert result["reason"] == "aggregate_byte_budget_exceeded"
    assert budget.written_bytes == 0
    assert not (tmp_path / "out").exists()
    assert (source / "big").read_bytes() == b"x" * 11


def test_multiple_files_share_quota_and_failed_tree_is_removed(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for name in ("one", "two"):
        (source / name).write_bytes(b"123456")
    budget = archive.ArchiveBudget(10)
    result = archive.copy_tree_hardlinked(source, tmp_path / "out", budget)
    assert not result["copied"]
    assert result["partial_files_removed"]
    assert budget.written_bytes == 6  # Failed cleanup never refunds write quota.
    assert not (tmp_path / "out").exists()


def test_wandb_and_hf_share_actual_copy_quota(tmp_path, monkeypatch):
    source = tmp_path / "repo" / "atropos" / "wandb"
    source.mkdir(parents=True)
    (source / "log").write_bytes(b"1234")
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "blob").write_bytes(b"5678901")
    monkeypatch.setattr(archive, "REPO_ROOT", tmp_path / "repo")
    monkeypatch.setenv("HF_HOME", str(cache))
    budget = archive.ArchiveBudget(10)
    assert archive.archive_local_wandb(tmp_path / "out", budget)["copied"]
    result = archive.archive_hf(tmp_path / "out", 200, budget)
    assert result["cache_copy"]["reason"] == "aggregate_byte_budget_exceeded"
    assert budget.written_bytes == 4


def test_deadline_before_copy_leaves_no_destination(tmp_path, monkeypatch):
    now = [0.0]
    monkeypatch.setattr(archive.time, "monotonic", lambda: now[0])
    budget = archive.ArchiveBudget(100, seconds=1)
    source = tmp_path / "source"
    source.mkdir()
    (source / "file").write_bytes(b"x")
    now[0] = 2
    result = archive.copy_tree_hardlinked(source, tmp_path / "out", budget)
    assert result["reason"] == "aggregate_deadline_exceeded"
    assert not (tmp_path / "out").exists()


def test_deadline_during_write_removes_temporary_file(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.write_bytes(b"x")
    budget = archive.ArchiveBudget(10)
    original = budget.write

    def expire(stream, data, **kwargs):
        original(stream, data, **kwargs)
        budget.deadline = 0
        budget.check()

    monkeypatch.setattr(budget, "write", expire)
    with pytest.raises(archive.BudgetExceeded, match="deadline"):
        archive.hardlink_or_copy(str(source), str(tmp_path / "out"), budget)
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob(".archive-*"))
    assert budget.written_bytes == 1


def test_remote_calls_fail_closed_even_with_credentials_and_small_estimates(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("remote provider/CLI must not run")

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(Api=forbidden))
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=forbidden))
    monkeypatch.setattr(archive, "run_cmd", forbidden)
    for key in ("WANDB_API_KEY", "TINKER_API_KEY", "HF_TOKEN"):
        monkeypatch.setenv(key, "synthetic-fixture")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "missing"))
    budget = archive.ArchiveBudget(100)
    wandb = archive.archive_wandb_cloud(tmp_path, ["fixture/project"], 0, 0, 0.001, budget)
    tinker = archive.archive_tinker(tmp_path, tmp_path, 100, 0, 0, budget)
    hf = archive.archive_hf(tmp_path, 200, budget)
    assert wandb["skipped"] == ["metadata", "history", "files"]
    assert wandb["reason"] == tinker["reason"] == hf["remote_metadata_note"] == "unsupported_bounded_remote_transfer"
    assert tinker["downloaded_count"] == 0
    assert budget.written_bytes == 0
    assert not list(tmp_path.iterdir())


def test_symlinks_and_special_files_cannot_escape_snapshot(tmp_path):
    src = tmp_path / "source"
    src.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("untouched")
    (src / "link").symlink_to(outside)
    result = archive.copy_tree_hardlinked(src, tmp_path / "out", archive.ArchiveBudget(100))
    assert result["copied"]
    assert not (tmp_path / "out" / "link").exists()
    assert outside.read_text() == "untouched"


def test_main_exhaustion_persists_counted_versioned_receipt(tmp_path, monkeypatch):
    source = tmp_path / "repo" / "atropos" / "wandb"
    source.mkdir(parents=True)
    (source / "big").write_bytes(b"x" * 100)
    root = tmp_path / "out"
    monkeypatch.setattr(archive, "REPO_ROOT", tmp_path / "repo")
    cap = archive.CONTROL_RESERVE_BYTES + 10
    monkeypatch.setattr(sys, "argv", ["archive", "--output-root", str(root), "--archive-budget-gb", str(cap / 1024**3), "--reserve-free-gb", "0", "--skip-hf"])
    assert archive.main() == 2
    receipt = json.loads((root / "MANIFEST.json").read_text())
    assert receipt["protocol"] == archive.ARCHIVE_PROTOCOL
    assert receipt["status"] == "partial_or_failed"
    assert receipt["budget"]["charged_write_bytes"] <= cap
    assert archive.dir_size_bytes(root) <= cap


@pytest.mark.parametrize("seconds", [0, -1, float("inf"), float("nan")])
def test_invalid_deadlines_fail_before_writes(seconds):
    with pytest.raises(ValueError):
        archive.ArchiveBudget(100, seconds)


def test_actual_bytes_override_understated_source_size(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.write_bytes(b"x" * 20)
    real_fstat = archive.os.fstat

    def understated(fd):
        info = real_fstat(fd)
        return SimpleNamespace(st_mode=info.st_mode, st_size=1)

    monkeypatch.setattr(archive.os, "fstat", understated)
    budget = archive.ArchiveBudget(10)
    with pytest.raises(archive.BudgetExceeded, match="byte_budget"):
        archive.hardlink_or_copy(str(source), str(tmp_path / "out"), budget)
    assert budget.written_bytes == 0
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob(".archive-*"))


def test_write_failure_cleans_temporary_and_preserves_previous_inode(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.write_bytes(b"new")
    target = tmp_path / "out"
    target.write_bytes(b"old")
    budget = archive.ArchiveBudget(100)

    def fail(*args, **kwargs):
        raise OSError("fixture write failure")

    monkeypatch.setattr(budget, "write", fail)
    with pytest.raises(OSError, match="fixture"):
        archive.hardlink_or_copy(str(source), str(target), budget)
    assert target.read_bytes() == b"old"
    assert source.read_bytes() == b"new"
    assert not list(tmp_path.glob(".archive-*"))


def test_redaction_replacement_obeys_same_budget(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "synthetic-fixture")
    source = tmp_path / "source"
    source.write_text("synthetic-fixture")
    root = tmp_path / "out"
    root.mkdir()
    budget = archive.ArchiveBudget(20)
    archive.hardlink_or_copy(str(source), str(root / "file.json"), budget)
    with pytest.raises(archive.BudgetExceeded):
        archive.redact_secrets(root, budget)
    assert not list(root.glob(".redacted-*"))
    assert source.read_text() == "synthetic-fixture"
    assert budget.written_bytes == len("synthetic-fixture")


def test_main_success_counts_receipt_itself_and_refuses_reuse(tmp_path, monkeypatch):
    monkeypatch.setattr(archive, "REPO_ROOT", tmp_path / "missing-repo")
    root = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["archive", "--output-root", str(root), "--archive-budget-gb", "0.001", "--reserve-free-gb", "0", "--skip-hf"])
    assert archive.main() == 0
    receipt = json.loads((root / "MANIFEST.json").read_text())
    assert receipt["status"] == "local_complete_remote_skipped"
    assert receipt["budget"]["charged_write_bytes"] == archive.dir_size_bytes(root)
    with pytest.raises(ValueError, match="fresh"):
        archive.main()
