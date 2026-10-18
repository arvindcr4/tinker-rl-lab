"""Mock HTTP only: compressed size, deadlines, target and secret-safe errors."""
from types import SimpleNamespace as NS
import urllib.request

import pytest
from platform_hybrid.skyrl import run_skyrl_tinker as trainer

URL = 'https://storage.googleapis.com/fixture/archive?X-Goog-Signature=SECRET'


class Response:
    def __init__(self, chunks, length=None):
        self.chunks = iter(chunks)
        self.headers = {} if length is None else {'Content-Length': str(length)}
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def read1(self, count):
        assert count == trainer.CHECKPOINT_DOWNLOAD_CHUNK_BYTES
        return next(self.chunks, b'')


def install(monkeypatch, response):
    def opener(*handlers):
        assert handlers[0].__class__.__name__ == 'NoRedirect'
        def open(url, timeout):
            assert timeout == trainer.CHECKPOINT_READ_TIMEOUT_SECONDS
            return response
        return NS(open=open)
    monkeypatch.setattr(urllib.request, 'build_opener', opener)


def test_streamed_download_without_content_length(tmp_path, monkeypatch):
    install(monkeypatch, Response([b'abc', b'def']))
    target = tmp_path/'archive'
    trainer.download_checkpoint_archive(URL, target)
    assert target.read_bytes() == b'abcdef'


@pytest.mark.parametrize('length', [None, 7])
def test_compressed_overflow_never_writes_beyond_cap(tmp_path, monkeypatch, length):
    monkeypatch.setattr(trainer, 'MAX_CHECKPOINT_DOWNLOAD_BYTES', 5)
    install(monkeypatch, Response([b'abc', b'def'], length))
    target = tmp_path/'archive'
    with pytest.raises(ValueError, match='byte/time'):
        trainer.download_checkpoint_archive(URL, target)
    assert not target.exists() or target.stat().st_size <= 5


def test_trickling_response_is_stopped_at_total_deadline(tmp_path, monkeypatch):
    install(monkeypatch, Response([b'a', b'b', b'c']))
    monkeypatch.setattr(trainer, 'MAX_CHECKPOINT_DOWNLOAD_SECONDS', 3)
    ticks = iter([0, 1, 2, 4])
    monkeypatch.setattr(trainer.time, 'monotonic', lambda: next(ticks))
    with pytest.raises(ValueError, match='byte/time'):
        trainer.download_checkpoint_archive(URL, tmp_path/'archive')


def test_io_timeout_does_not_expose_signed_query(tmp_path, monkeypatch):
    def opener(*args):
        def open(*args, **kwargs): raise TimeoutError(URL)
        return NS(open=open)
    monkeypatch.setattr(urllib.request, 'build_opener', opener)
    with pytest.raises(ValueError) as error:
        trainer.download_checkpoint_archive(URL, tmp_path/'archive')
    assert 'SECRET' not in str(error.value) and error.value.__suppress_context__


@pytest.mark.parametrize('url', [
    'file:///tmp/archive', 'http://storage.googleapis.com/archive',
    'https://storage.googleapis.com.evil.invalid/archive',
    'https://user:secret@storage.googleapis.com/archive',
    'https://storage.googleapis.com:8443/archive', 'https://127.0.0.1/archive',
])
def test_invalid_targets_rejected_before_http(tmp_path, monkeypatch, url):
    monkeypatch.setattr(urllib.request, 'build_opener', lambda *a: pytest.fail('must not contact target'))
    with pytest.raises(ValueError, match='approved HTTPS'):
        trainer.download_checkpoint_archive(url, tmp_path/'archive')


def test_redirect_handler_fails_closed(tmp_path, monkeypatch):
    def opener(handler):
        def open(*args, **kwargs):
            return handler.redirect_request(None, None, 302, '', {}, 'https://evil.invalid/?SECRET')
        return NS(open=open)
    monkeypatch.setattr(urllib.request, 'build_opener', opener)
    with pytest.raises(ValueError) as error:
        trainer.download_checkpoint_archive(URL, tmp_path/'archive')
    assert 'SECRET' not in str(error.value)
