"""Synthetic archive and mocked publication tests; no network/authentication."""

import io
import shutil
import sys
import tarfile
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS

import pytest
from platform_hybrid.skyrl import run_skyrl_tinker as trainer


def archive(path, entries, gzip=True):
    with tarfile.open(path, "w:gz" if gzip else "w") as tar:
        for name, kind, data in entries:
            member = tarfile.TarInfo(name)
            member.type = kind
            if kind in (tarfile.LNKTYPE, tarfile.SYMTYPE):
                member.linkname = "../outside"
            member.size = len(data) if kind == tarfile.REGTYPE else 0
            tar.addfile(member, io.BytesIO(data) if member.isfile() else None)


@pytest.mark.parametrize(
    ("name", "kind"),
    [
        ("/tmp/absolute", tarfile.REGTYPE),  # noqa: S108 - Deliberately unsafe archive entry; extraction must reject it.
        ("../outside", tarfile.REGTYPE),
        ("weights/../../outside", tarfile.REGTYPE),
        ("link", tarfile.LNKTYPE),
        ("link", tarfile.SYMTYPE),
        ("device", tarfile.CHRTYPE),
        ("pipe", tarfile.FIFOTYPE),
    ],
)
def test_reject_entire_unsafe_inventory_before_any_extraction(tmp_path, name, kind):
    source = tmp_path / "weights.tar"
    target = tmp_path / "target"
    target.mkdir()
    archive(source, [("valid.bin", tarfile.REGTYPE, b"valid"), (name, kind, b"bad")])
    with pytest.raises(ValueError, match="unsafe"):
        trainer.extract_checkpoint_archive(source, target)
    assert not list(target.iterdir())


@pytest.mark.parametrize(
    ("limit", "value"), [("MAX_CHECKPOINT_MEMBERS", 1), ("MAX_CHECKPOINT_BYTES", 3)]
)
def test_archive_limits_precede_extraction(tmp_path, monkeypatch, limit, value):
    source = tmp_path / "weights.tar"
    target = tmp_path / "target"
    target.mkdir()
    archive(source, [("one", tarfile.REGTYPE, b"aa"), ("two", tarfile.REGTYPE, b"bb")])
    monkeypatch.setattr(trainer, limit, value)
    with pytest.raises(ValueError, match="limit"):
        trainer.extract_checkpoint_archive(source, target)
    assert not list(target.iterdir())


@pytest.mark.parametrize(("valid", "gzip"), [(False, True), (True, True), (True, False)])
def test_full_upload_validates_before_publication_and_cleans_owned_staging(
    tmp_path, monkeypatch, valid, gzip
):
    source = tmp_path / "input.tar"
    victim = tmp_path / "untouched"
    victim.write_text("original")
    archive(source, [("adapter.bin" if valid else str(victim), tarfile.REGTYPE, b"weights")], gzip)
    events = []
    staging = []
    hub = ModuleType("huggingface_hub")

    def upload(**kwargs):
        path = Path(kwargs["folder_path"])
        assert (path / "adapter.bin").read_bytes() == b"weights"
        assert not (path / "checkpoint.tar.gz").exists()
        events.append("upload")

    hub.HfApi = lambda: NS(create_repo=lambda **k: events.append("create"), upload_folder=upload)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    def download(url, dest):
        staging.append(Path(dest).parent)
        assert Path(dest).parent.stat().st_mode & 0o777 == 0o700
        shutil.copyfile(source, dest)

    monkeypatch.setattr(trainer, "download_checkpoint_archive", download)
    svc = NS(
        create_rest_client=lambda: NS(
            get_checkpoint_archive_url_from_tinker_path=lambda _: NS(
                result=lambda: NS(url="https://fixture.invalid/archive")
            )
        )
    )
    cfg = dict(
        huggingface=dict(push_to_hub=True, model_card=False, hub_org="fixture"),
        model=dict(name="mock"),
    )
    call = lambda: trainer.upload_to_hub(cfg, "fixture", svc, "checkpoint", "run", {})
    if valid:
        assert call().startswith("https://huggingface.co/fixture/")
        assert events == ["create", "upload"]
    else:
        with pytest.raises(ValueError, match="unsafe"):
            call()
        assert events == []
    assert victim.read_text() == "original"
    assert staging
    assert all(not path.exists() for path in staging)


def test_duplicate_archive_member_rejected(tmp_path):
    source = tmp_path / "weights.tar"
    target = tmp_path / "target"
    target.mkdir()
    archive(source, [("same.bin", tarfile.REGTYPE, b"one"), ("same.bin", tarfile.REGTYPE, b"two")])
    with pytest.raises(ValueError, match="duplicate"):
        trainer.extract_checkpoint_archive(source, target)
    assert not list(target.iterdir())
