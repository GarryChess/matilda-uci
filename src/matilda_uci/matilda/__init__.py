"""Matilda inference core: the set-transformer that re-ranks frozen Maia-3.

Faithful port of the paper pipeline (github.com/GarryChess/matilda1-paper,
read-only): :mod:`.model` is ``TXTC`` from ``pipeline/train/train_base_tc.py``
with the widened candidate projection the shipped checkpoints use, and
:mod:`.features` is ``tensors_sf`` from ``pipeline/train/train_sf_smoke.py``.
Numerical parity with the paper implementation is asserted by
``scripts/verify_checkpoint.py``.

Imported only when the Matilda backend is actually used, so the rest of
``matilda_uci`` (the UCI engine, the :class:`~matilda_uci.policy.MovePolicy`
protocol) stays importable and testable without ``torch``.
"""

from __future__ import annotations

from .features import SF_CP_UNSCORED, tensors_sf
from .inference import Maia3FeatureError, MatildaModel, MatildaPrediction
from .search import (
    CP_UNSCORED,
    MATE_SCORE,
    DumbSearchController,
    MoveScore,
    SearchController,
    UciSearchController,
    scores_to_arrays,
)
from .model import (
    D_MODEL,
    FD_BASE,
    FD_CAND,
    FD_SF,
    HIDDEN_DIM,
    IMPORTANCE_DIM,
    MOVE_EMB_DIM,
    N_CANDIDATES,
    STYLE_DIM,
    TC_DIM,
    VOCAB_SIZE,
    TXTC,
    maia3_reference_logprob,
)

__all__ = [
    "TXTC",
    "MatildaModel",
    "MatildaPrediction",
    "Maia3FeatureError",
    "SearchController",
    "UciSearchController",
    "DumbSearchController",
    "MoveScore",
    "scores_to_arrays",
    "MATE_SCORE",
    "CP_UNSCORED",
    "tensors_sf",
    "maia3_reference_logprob",
    "SF_CP_UNSCORED",
    "VOCAB_SIZE",
    "HIDDEN_DIM",
    "IMPORTANCE_DIM",
    "N_CANDIDATES",
    "D_MODEL",
    "FD_BASE",
    "FD_SF",
    "FD_CAND",
    "MOVE_EMB_DIM",
    "TC_DIM",
    "STYLE_DIM",
]
