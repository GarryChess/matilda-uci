from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MaiaResult:
    """Maia-2 output for one board state."""

    move_probs: dict[str, float]  # uci -> probability (real board orientation)
    win_prob: float
    top_moves: tuple[tuple[str, float], ...]  # (uci, prob) sorted desc, len <= top_k
    embedding: np.ndarray | None = field(default=None)  # penultimate activation

    def logit(self, uci: str, *, floor: float = 1e-9) -> float:
        return math.log(max(self.move_probs.get(uci, 0.0), floor))


class MaiaWrapper:
    """Wraps real Maia-2 inference behind a clean interface.

    The heavy ``maia2`` package is imported lazily so the rest of the package
    (and the unit tests, which inject a fake wrapper) stays importable without it.
    """

    def __init__(
        self,
        *,
        maia_type: str = "rapid",
        device: str = "cpu",
        top_k: int = 5,
        capture_embedding: bool = False,
    ) -> None:
        self.maia_type = maia_type
        self.device = device
        self.top_k = top_k
        self.capture_embedding = capture_embedding
        self._model = None
        self._prepared = None
        self._embedding_buffer: np.ndarray | None = None
        self._hook_handle = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from maia2 import inference, model
        except ImportError as exc:  # pragma: no cover - depends on optional dep
            raise ImportError(
                "maia2 is required for MaiaWrapper. It is a dependency of matilda-uci: pip install matilda-uci"
            ) from exc

        self._model = model.from_pretrained(type=self.maia_type, device=self.device)
        self._prepared = inference.prepare()
        if self.capture_embedding:
            self._register_embedding_hook()

    def _register_embedding_hook(self) -> None:
        """Best-effort forward hook to capture the pre-logits activation.

        Maia-2 does not expose the penultimate embedding publicly, so we attach
        a hook to the module feeding the policy head. If we cannot locate it the
        env falls back to a zero vector (documented Week-1 limitation).
        """
        import torch.nn as nn

        target = None
        # Heuristic: the last nn.Linear before the policy output, or a module
        # explicitly named like a fully-connected/policy layer.
        for name, module in self._model.named_modules():
            lname = name.lower()
            if isinstance(module, nn.Linear) and ("fc" in lname or "policy" in lname or "head" in lname):
                target = module
        if target is None:
            logger.warning(
                "MaiaWrapper: could not locate a penultimate layer for embeddings; "
                "embeddings will be zeros."
            )
            return

        def _hook(_module, inputs, _output):
            if inputs and inputs[0] is not None:
                self._embedding_buffer = inputs[0].detach().cpu().numpy().reshape(-1)

        self._hook_handle = target.register_forward_hook(_hook)

    def infer(self, board: "object", *, elo_self: int, elo_oppo: int) -> MaiaResult:
        from maia2 import inference

        self._ensure_loaded()
        self._embedding_buffer = None
        move_probs, win_prob = inference.inference_each(
            self._model, self._prepared, board.fen(), int(elo_self), int(elo_oppo)
        )
        # Verified (maia2==0.9): keys are real-board UCIs for both colors, so no
        # mirroring is needed. The env restricts these to legal moves downstream.
        move_probs = {str(k): float(v) for k, v in move_probs.items()}
        top_moves = tuple(
            sorted(move_probs.items(), key=lambda kv: kv[1], reverse=True)[: self.top_k]
        )
        embedding = self._embedding_buffer if self.capture_embedding else None
        return MaiaResult(
            move_probs=move_probs,
            win_prob=float(win_prob),
            top_moves=top_moves,
            embedding=embedding,
        )

    def close(self) -> None:
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
