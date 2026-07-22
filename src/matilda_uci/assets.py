"""Checkpoint resolution: local path, repo checkout, user cache, or download.

``pip install matilda-uci`` ships no weights; the released re-ranker
checkpoints are GitHub release assets, fetched once into the user cache so
the default ``--checkpoint`` works with no checkout and no flags. A git
clone needs no download at all: the same files are vendored in
``checkpoints/`` via git-lfs.

Lookup order for a known name like ``base_3k.pt`` (bare or prefixed
``checkpoints/``): the path as given, the checkout's ``checkpoints/``
directory, the cache (``$MATILDA_UCI_CACHE`` > ``$XDG_CACHE_HOME`` >
``~/.cache``, under ``matilda-uci/``), then the release download. Downloads
are sha256-verified and written atomically.
"""

from __future__ import annotations

import hashlib
import logging
import os
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

_RELEASE_URL = (
    "https://github.com/GarryChess/matilda-uci/releases/download/"
    "checkpoints-v1/{name}"
)

# name -> sha256 of the released artifact
KNOWN_CHECKPOINTS = {
    "base_3k.pt":
        "271aece522a785f3f4e5dfd7f17939ce8f83a3f7a00ee6d456534971ecb55acf",
    "style_token_3k.pt":
        "58d5875a6ef1833a79acfb1e24347d7253882c042d1f887b654bb0208527f513",
    "maia3_zero.pt":
        "5aef9c16cdfb9e3aa7c850c3c4ee438a697dafa44afe7b307ecf779b7ae0f704",
}


def cache_dir() -> Path:
    override = os.environ.get("MATILDA_UCI_CACHE")
    if override:
        return Path(override)
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "matilda-uci"


def _repo_checkpoints() -> Path:
    # src/matilda_uci/assets.py -> repo root /checkpoints (a checkout only;
    # for a site-packages install this simply won't exist).
    return Path(__file__).resolve().parents[2] / "checkpoints"


def resolve_checkpoint(spec: str | os.PathLike, *, download: bool = True) -> str:
    """Resolve a checkpoint argument to a real file path.

    ``spec`` is either a path to an existing file (returned as given) or one
    of the released checkpoint names in :data:`KNOWN_CHECKPOINTS`. Raises
    ``FileNotFoundError`` for anything else, and ``RuntimeError`` when a
    download fails or does not match its pinned sha256.
    """
    path = Path(spec)
    if path.is_file():
        _reject_lfs_pointer(path)
        return str(path)
    name = path.name
    if name not in KNOWN_CHECKPOINTS:
        raise FileNotFoundError(
            f"checkpoint not found: {str(spec)!r} — not an existing file, and "
            f"not one of the released names ({', '.join(sorted(KNOWN_CHECKPOINTS))})"
        )
    repo_copy = _repo_checkpoints() / name
    if repo_copy.is_file():
        _reject_lfs_pointer(repo_copy)
        return str(repo_copy)
    cached = cache_dir() / name
    if cached.is_file():
        return str(cached)
    if not download:
        raise FileNotFoundError(
            f"checkpoint {name!r} is not present locally (download disabled)"
        )
    return str(_download(name, cached))


def resolve_if_released(spec: str | os.PathLike, *, download: bool = True) -> str:
    """Lenient sibling of :func:`resolve_checkpoint` for library entry points.

    An existing file, or any caller-supplied identifier that is not a released
    name (e.g. a stand-in for an injected model), passes through untouched;
    only a released name that is not already on disk gets resolved (and, on
    first use, downloaded). This lets ``MatildaModel()`` / ``MatildaPolicy()``
    work straight after ``pip install`` without turning every custom string
    into a hard error.
    """
    path = Path(spec)
    if not path.is_file() and path.name in KNOWN_CHECKPOINTS:
        return resolve_checkpoint(spec, download=download)
    return str(spec)


def _reject_lfs_pointer(path: Path) -> None:
    """A clone made without git-lfs leaves ~130-byte pointer stubs where the
    checkpoints should be; loading one into torch fails inscrutably, so name
    the problem here instead."""
    if path.stat().st_size < 512:
        with open(path, "rb") as fh:
            if fh.read(28).startswith(b"version https://git-lfs"):
                raise RuntimeError(
                    f"{path} is a git-lfs pointer stub, not the checkpoint — "
                    "install git-lfs (https://git-lfs.com) and run "
                    "`git lfs pull` in the checkout"
                )


def _download(name: str, dest: Path) -> Path:
    url = _RELEASE_URL.format(name=name)
    logger.info("downloading %s -> %s", url, dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(url) as resp, open(tmp, "wb") as out:
            while chunk := resp.read(1 << 20):
                digest.update(chunk)
                out.write(chunk)
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"could not download checkpoint {name!r} from {url}: {exc} — "
            "if you are offline, pass --checkpoint /path/to/the/file instead"
        ) from exc
    if digest.hexdigest() != KNOWN_CHECKPOINTS[name]:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"downloaded checkpoint {name!r} failed its sha256 check "
            "(corrupted or tampered download; try again)"
        )
    os.replace(tmp, dest)  # atomic: never leaves a half-written checkpoint
    return dest
