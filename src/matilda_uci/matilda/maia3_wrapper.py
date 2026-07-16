"""Maia-3 inference behind a clean interface for the RL env.

Loads the raw PyTorch Chessformer (not the ``maia3-uci`` engine) so we can expose
research-grade signals the UCI interface hides:

- the **full policy logits** over the 4352-move vocabulary (not just top-k),
- an optional **pre-logits hidden layer** (the pooled penultimate activation), and
- an optional **GAB per-square importance map** (8x8), derived from a transformer
  block's Geometric Attention Bias.

All heavy ``maia3``/``torch`` imports are lazy so the package stays importable and
testable (the env injects a fake wrapper in tests). Predictions are produced in
Maia's side-to-move frame (board mirrored for Black); ``move_probs`` and the
importance map are returned in real-board orientation.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# Maia-3 move vocabulary size: 64*64 from->to + 256 promotions.
MAIA3_VOCAB_SIZE = 4352


@dataclass(frozen=True)
class Maia3Result:
    """Maia-3 outputs for one board state."""

    move_probs: dict[str, float]  # real-board uci -> prob (legal moves, normalized)
    logits: np.ndarray  # full (4352,) policy logits in Maia's side-to-move frame
    win_prob: float  # expected score in [0,1] = P(win) + 0.5*P(draw)
    top_moves: tuple[tuple[str, float], ...]  # (uci, prob) desc, len <= top_k
    hidden: np.ndarray | None = field(default=None)  # pooled pre-logits activation
    importance: np.ndarray | None = field(default=None)  # (8,8) real-board importance
    ponder: float | None = field(default=None)  # Maia-3 think-time head output

    def logit(self, uci: str, *, floor: float = 1e-9) -> float:
        return math.log(max(self.move_probs.get(uci, 0.0), floor))


class Maia3Wrapper:
    """Wraps real Maia-3 inference; see module docstring for the exposed signals."""

    def __init__(
        self,
        *,
        model: str = "23m",
        device: str = "cpu",
        top_k: int = 5,
        capture_hidden: bool = False,
        capture_importance: bool = False,
        importance_block: int = -1,
    ) -> None:
        self.model = model
        self.device = device
        self.top_k = top_k
        self.capture_hidden = capture_hidden
        self.capture_importance = capture_importance
        self.importance_block = importance_block

        self._cfg = None
        self._model = None
        self._move_to_idx: dict[str, int] = {}
        self._idx_to_move: dict[int, str] = {}
        self._hidden_buffer = None
        self._query_buffer = None
        self._hooks: list = []

    # --- loading ---------------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from types import SimpleNamespace

            from maia3.model_registry import (
                apply_model_config,
                resolve_checkpoint_path,
                resolve_model_spec,
            )
            from maia3.uci import load_model
            from maia3.utils import get_all_possible_moves
        except ImportError as exc:  # pragma: no cover - depends on optional dep
            raise ImportError(
                "maia3 is required for Maia3Wrapper. Install the paper's pinned "
                "revision: pip install 'maia3 @ git+https://github.com/CSSLab/"
                "maia3.git@1e13597c42d4858b7cfd7cfdae01e297263364b2' "
                "(weights auto-download from HuggingFace on first use)."
            ) from exc

        spec = resolve_model_spec(self.model)
        cfg = SimpleNamespace(device=self.device, use_amp=False, use_uci_history=False)
        apply_model_config(cfg, spec)
        cfg.checkpoint_path = resolve_checkpoint_path(spec)
        self._cfg = cfg
        self._model = load_model(cfg)  # builds MAIA3Model, loads weights, eval()

        moves = get_all_possible_moves()
        self._move_to_idx = {m: i for i, m in enumerate(moves)}
        self._idx_to_move = dict(enumerate(moves))

        if self.capture_hidden:
            self._register_hidden_hook()
        if self.capture_importance:
            self._register_importance_hook()

    def _register_hidden_hook(self) -> None:
        """Capture the pooled pre-logits activation (LayerNorm feeding the heads)."""
        last_ln = getattr(self._model, "last_ln", None)
        if last_ln is None:  # pragma: no cover - architecture guard
            logger.warning("Maia3Wrapper: no last_ln module; hidden will be None.")
            return

        def _hook(_module, _inputs, output):
            self._hidden_buffer = output.detach().cpu().numpy().reshape(-1)

        self._hooks.append(last_ln.register_forward_hook(_hook))

    def _register_importance_hook(self) -> None:
        """Capture the query into a chosen block's attention to recompute GAB bias."""
        try:
            block = self._model.transformer.layers[self.importance_block]
            mha = block.self_attn
        except (AttributeError, IndexError):  # pragma: no cover - architecture guard
            logger.warning("Maia3Wrapper: could not locate GAB block; importance=None.")
            return
        if not getattr(mha, "use_gab", False):
            logger.warning("Maia3Wrapper: model has no GAB; importance will be None.")
            return

        def _pre_hook(_module, args, kwargs):
            q = kwargs.get("query", args[0] if args else None)
            if q is not None:
                self._query_buffer = q.detach()

        self._hooks.append(mha.register_forward_pre_hook(_pre_hook, with_kwargs=True))
        self._importance_mha = mha

    # --- inference -------------------------------------------------------------
    def infer(
        self,
        board: "object",
        *,
        elo_self: int,
        elo_oppo: int,
        board_history: "list[str] | None" = None,
    ) -> Maia3Result:
        """Run Maia-3 on ``board``.

        ``board_history`` is the real sequence of board FENs that preceded this
        position in the game (oldest -> newest, **excluding** ``board``). Maia-3
        consumes the most recent ``cfg.history`` board states; passing the true
        game history (rather than padding from the current board) is what lets it
        condition on how the player actually reached the position. Each board is
        tokenized in its own side-to-move frame, matching Maia-3's convention.
        """
        import chess
        import torch
        from collections import deque

        from maia3.dataset import (
            get_historical_tokens,
            get_legal_moves_mask,
            tokenize_board,
        )
        from maia3.utils import mirror_move

        self._ensure_loaded()
        cfg = self._cfg
        self._hidden_buffer = None
        self._query_buffer = None

        boards = []
        for fen in board_history or []:
            if not fen:
                continue
            try:
                boards.append(tokenize_board(chess.Board(fen)))
            except ValueError:
                continue
        boards.append(tokenize_board(board))  # current position last (newest)
        history = deque(boards, maxlen=cfg.history)
        tokens = get_historical_tokens(
            history, cfg, base=0.0, inc=0.0, clk_left_before=0.0, clk_ponder=0.0
        )
        tokens = tokens.unsqueeze(0).to(cfg.device)
        self_elos = torch.tensor([int(elo_self)], dtype=torch.long, device=cfg.device)
        oppo_elos = torch.tensor([int(elo_oppo)], dtype=torch.long, device=cfg.device)

        with torch.no_grad():
            logits_move, logits_value, logits_ponder = self._model(
                tokens, self_elos, oppo_elos
            )

        logits = logits_move[0].float()
        legal_mask = get_legal_moves_mask(board, self._move_to_idx).to(logits.device)
        masked = logits.masked_fill(~legal_mask, float("-inf"))
        probs = torch.softmax(masked, dim=-1)

        # Map legal-move probabilities back to real-board UCIs.
        is_black = board.turn == chess.BLACK
        move_probs: dict[str, float] = {}
        for idx in torch.nonzero(legal_mask, as_tuple=False).flatten().tolist():
            uci = self._idx_to_move[idx]
            if is_black:
                uci = mirror_move(uci)
            move_probs[uci] = float(probs[idx])
        top_moves = tuple(
            sorted(move_probs.items(), key=lambda kv: kv[1], reverse=True)[: self.top_k]
        )

        loss, draw, win = torch.softmax(logits_value[0].float(), dim=-1).tolist()
        win_prob = float(win + 0.5 * draw)
        ponder = float(logits_ponder.reshape(-1)[0]) if logits_ponder is not None else None

        hidden = (
            np.asarray(self._hidden_buffer, dtype=np.float32)
            if self._hidden_buffer is not None
            else None
        )
        importance = self._compute_importance(is_black) if self.capture_importance else None

        return Maia3Result(
            move_probs=move_probs,
            logits=logits.detach().cpu().numpy().astype(np.float32),
            win_prob=win_prob,
            top_moves=top_moves,
            hidden=hidden,
            importance=importance,
            ponder=ponder,
        )

    def _compute_importance(self, is_black: bool) -> "np.ndarray | None":
        """Reduce the captured GAB square-pair bias to an 8x8 real-board map."""
        if self._query_buffer is None or getattr(self, "_importance_mha", None) is None:
            return None
        try:
            import torch

            with torch.no_grad():
                bias = self._importance_mha._sq_bias(self._query_buffer)  # (1,H,64,64)
                # Per source-square importance: mean over heads and partner squares.
                per_square = bias[0].mean(dim=0).mean(dim=-1)  # (64,)
                values = per_square.detach().cpu().numpy().astype(np.float32)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Maia3Wrapper: importance computation failed: %s", exc)
            return None

        # Maia squares are in side-to-move frame (vertically mirrored for Black);
        # un-mirror to real-board orientation so the heatmap matches the FEN.
        if is_black:
            import chess

            values = values[[chess.square_mirror(s) for s in range(64)]]
        # square index s = rank*8 + file -> row 0 is rank 1 (a1..h1).
        grid = values.reshape(8, 8)
        lo, hi = float(grid.min()), float(grid.max())
        if hi > lo:
            grid = (grid - lo) / (hi - lo)  # normalize to [0,1] for a clean heatmap
        return grid.astype(np.float32)

    def close(self) -> None:
        for handle in self._hooks:
            handle.remove()
        self._hooks = []
