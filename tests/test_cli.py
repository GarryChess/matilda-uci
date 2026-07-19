"""CLI fail-fast validation tests (torch-free: validation precedes any load).

A config error that survives startup surfaces as an endless ``bestmove 0000``
null-move loop at the first ``go`` — so bad flags must die in argparse.
"""

from __future__ import annotations

import pytest

from matilda_uci.cli import build_parser, validate_args


def _validate(argv: list[str]) -> None:
    parser = build_parser()
    validate_args(parser, parser.parse_args(argv))


def test_bad_device_for_matilda_backend_is_rejected() -> None:
    with pytest.raises(SystemExit):
        _validate(["--device", "gpu"])  # gpu is a maia2-ism; matilda wants cpu/mps/cuda


def test_bad_device_for_maia2_backend_is_rejected() -> None:
    with pytest.raises(SystemExit):
        _validate(["--backend", "maia2", "--device", "mps"])


def test_maia_type_under_matilda_backend_is_rejected_not_ignored() -> None:
    with pytest.raises(SystemExit):
        _validate(["--maia-type", "blitz"])  # silently discarding it misled users


def test_missing_checkpoint_is_rejected_at_startup(tmp_path) -> None:
    with pytest.raises(SystemExit):
        _validate(["--checkpoint", str(tmp_path / "nope.pt")])


def test_missing_style_checkpoint_is_rejected(tmp_path) -> None:
    ckpt = tmp_path / "base.pt"
    ckpt.write_bytes(b"x")  # existence is all validate checks
    with pytest.raises(SystemExit):
        _validate(
            ["--checkpoint", str(ckpt), "--style-checkpoint", str(tmp_path / "no.pt")]
        )


def test_style_vector_requires_style_checkpoint(tmp_path) -> None:
    ckpt = tmp_path / "base.pt"
    vec = tmp_path / "vec.pt"
    ckpt.write_bytes(b"x")
    vec.write_bytes(b"x")
    with pytest.raises(SystemExit):
        _validate(["--checkpoint", str(ckpt), "--style-vector", str(vec)])


def test_maia2_backend_accepts_maia_type_and_gpu() -> None:
    _validate(["--backend", "maia2", "--maia-type", "blitz", "--device", "gpu"])


def _fake_runtime(monkeypatch, *, stockfish: str | None) -> None:
    """Pretend the maia3 package is importable and stockfish is (or isn't)
    on PATH, so validation success paths don't need either installed (CI)."""
    import matilda_uci.cli as cli

    monkeypatch.setattr(cli.shutil, "which", lambda name: stockfish)
    monkeypatch.setattr(cli.importlib.util, "find_spec", lambda name: object())


def test_engine_required_by_default(monkeypatch, tmp_path) -> None:
    ckpt = tmp_path / "base.pt"
    ckpt.write_bytes(b"x")
    _fake_runtime(monkeypatch, stockfish=None)
    with pytest.raises(SystemExit):
        _validate(["--checkpoint", str(ckpt)])  # no stockfish -> startup error
    _validate(["--checkpoint", str(ckpt), "--no-engine"])  # explicit opt-out
    _fake_runtime(monkeypatch, stockfish="/fake/bin/stockfish")
    _validate(["--checkpoint", str(ckpt)])  # auto-resolution succeeds


def test_no_engine_conflicts_with_engine_cmd(monkeypatch, tmp_path) -> None:
    ckpt = tmp_path / "base.pt"
    ckpt.write_bytes(b"x")
    _fake_runtime(monkeypatch, stockfish="/fake/bin/stockfish")
    with pytest.raises(SystemExit):
        _validate(["--checkpoint", str(ckpt), "--no-engine",
                   "--engine-cmd", "stockfish"])


def test_explicit_engine_cmd_must_exist(monkeypatch, tmp_path) -> None:
    ckpt = tmp_path / "base.pt"
    ckpt.write_bytes(b"x")
    _fake_runtime(monkeypatch, stockfish=None)
    with pytest.raises(SystemExit):
        _validate(["--checkpoint", str(ckpt), "--engine-cmd", "no-such-engine"])
    _fake_runtime(monkeypatch, stockfish="/fake/bin/lc0")
    _validate(["--checkpoint", str(ckpt), "--engine-cmd", "lc0 --weights=w.pb"])


def test_opp_elo_defaults_to_elo(tmp_path) -> None:
    from matilda_uci.cli import build_policy

    ckpt = tmp_path / "base.pt"
    ckpt.write_bytes(b"x")
    parser = build_parser()
    args = parser.parse_args(
        ["--elo", "2400", "--checkpoint", str(ckpt), "--no-engine"]
    )
    policy = build_policy(args)
    assert policy.elo_oppo == 2400  # follows --elo when not given
    args = parser.parse_args(
        ["--elo", "2400", "--opp-elo", "1800", "--checkpoint", str(ckpt),
         "--no-engine"]
    )
    assert build_policy(args).elo_oppo == 1800  # explicit value wins
