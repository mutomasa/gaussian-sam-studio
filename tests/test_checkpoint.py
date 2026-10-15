import pytest

from spatialsam.models.checkpoint import SAM31_CHECKPOINT, resolve_checkpoint


def test_explicit_path_wins():
    def fail(**_):
        raise AssertionError("ダウンロードしない")

    assert resolve_checkpoint("/w.pt", "facebook/sam3.1", "abc", download=fail) == "/w.pt"


def test_downloads_pinned_revision():
    calls = []
    path = resolve_checkpoint(None, "facebook/sam3.1", "abc", download=lambda **kw: calls.append(kw) or "/c.pt")
    assert path == "/c.pt"
    assert calls == [{"repo_id": "facebook/sam3.1", "filename": SAM31_CHECKPOINT, "revision": "abc"}]


def test_requires_source_without_path():
    with pytest.raises(ValueError):
        resolve_checkpoint(None, "facebook/sam3.1", "")
