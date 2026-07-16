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


def test_maia2_backend_accepts_maia_type_and_gpu() -> None:
    _validate(["--backend", "maia2", "--maia-type", "blitz", "--device", "gpu"])
