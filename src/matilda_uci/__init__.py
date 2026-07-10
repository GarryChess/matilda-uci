"""Matilda: a free human-like UCI chess engine.

Exposes a rating-conditioned human-move model (Maia-2 today) over the standard
UCI protocol, so it plugs into any chess GUI or engine runner. The
:class:`MovePolicy` protocol is the deliberate swap seam: future human-like
backends (Maia-3, style-conditioned models) implement the same surface and slot
into the engine unchanged.
"""

from __future__ import annotations

from .engine import UciEngine
from .policy import MaiaPolicy, MovePolicy, PolicyResult, UciOption, cp_to_winprob, winprob_to_cp

__version__ = "0.1.0"

__all__ = [
    "UciEngine",
    "MaiaPolicy",
    "MovePolicy",
    "PolicyResult",
    "UciOption",
    "winprob_to_cp",
    "cp_to_winprob",
    "__version__",
]
