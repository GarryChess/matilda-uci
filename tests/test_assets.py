"""Checkpoint resolution and the sha256-verified first-run download."""

from __future__ import annotations

import hashlib

import pytest

import matilda_uci.assets as assets
from matilda_uci.assets import resolve_checkpoint


@pytest.fixture()
def isolated(monkeypatch, tmp_path):
    """No repo checkpoints, an empty cache, and a local 'release' to download
    from; returns (cache_dir, release_dir)."""
    cache = tmp_path / "cache"
    release = tmp_path / "release"
    release.mkdir()
    monkeypatch.chdir(tmp_path)  # hide the real checkout's checkpoints/
    monkeypatch.setattr(assets, "_repo_checkpoints", lambda: tmp_path / "norepo")
    monkeypatch.setenv("MATILDA_UCI_CACHE", str(cache))
    monkeypatch.setattr(assets, "_RELEASE_URL",
                        release.as_uri() + "/{name}")
    return cache, release


def _publish(release, name: str, payload: bytes, monkeypatch) -> None:
    (release / name).write_bytes(payload)
    monkeypatch.setitem(assets.KNOWN_CHECKPOINTS, name,
                        hashlib.sha256(payload).hexdigest())


def test_existing_path_returned_as_given(tmp_path) -> None:
    f = tmp_path / "anything.pt"
    f.write_bytes(b"x")
    assert resolve_checkpoint(str(f)) == str(f)


def test_unknown_missing_name_raises(isolated) -> None:
    with pytest.raises(FileNotFoundError, match="not one of the released"):
        resolve_checkpoint("no_such_checkpoint.pt")


def test_download_verify_and_cache(isolated, monkeypatch) -> None:
    cache, release = isolated
    _publish(release, "base_3k.pt", b"checkpoint-bytes", monkeypatch)
    path = resolve_checkpoint("checkpoints/base_3k.pt")  # prefix form works
    assert path == str(cache / "base_3k.pt")
    assert (cache / "base_3k.pt").read_bytes() == b"checkpoint-bytes"
    # second resolve is a pure cache hit (kill the release to prove it)
    (release / "base_3k.pt").unlink()
    assert resolve_checkpoint("base_3k.pt") == path


def test_download_rejects_bad_sha256(isolated, monkeypatch) -> None:
    cache, release = isolated
    _publish(release, "base_3k.pt", b"good", monkeypatch)
    (release / "base_3k.pt").write_bytes(b"evil")  # content != pinned digest
    with pytest.raises(RuntimeError, match="sha256"):
        resolve_checkpoint("base_3k.pt")
    assert not (cache / "base_3k.pt").exists()  # nothing half-written kept


def test_download_disabled_raises(isolated) -> None:
    with pytest.raises(FileNotFoundError, match="download disabled"):
        resolve_checkpoint("base_3k.pt", download=False)


def test_lfs_pointer_stub_named_not_loaded(tmp_path) -> None:
    stub = tmp_path / "base_3k.pt"
    stub.write_bytes(
        b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 1\n"
    )
    with pytest.raises(RuntimeError, match="git-lfs pointer"):
        resolve_checkpoint(str(stub))


def test_repo_checkout_wins_before_cache(isolated, monkeypatch, tmp_path) -> None:
    repo = tmp_path / "repo-ckpts"
    repo.mkdir()
    (repo / "base_3k.pt").write_bytes(b"from-checkout")
    monkeypatch.setattr(assets, "_repo_checkpoints", lambda: repo)
    assert resolve_checkpoint("base_3k.pt") == str(repo / "base_3k.pt")
