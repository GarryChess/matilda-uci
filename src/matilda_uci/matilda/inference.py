"""MatildaModel: one object that runs the full inference chain.

    FEN + history --Maia-3 (frozen, 23M)--> logits/hidden/attention/candidates
                  --featurize (paper-exact)--> TXTC inputs
                  --TXTC (base_3k.pt)--> re-ranked human move distribution

Featurization mirrors ``pipeline/extract/featurize_maia3_only.py`` from the
paper repo exactly, including the fp16 round-trip the training shards baked in
(features were stored as float16, so the model was trained on fp16-quantized
Maia-3 outputs — we quantize identically). The optional engine block comes from
a :class:`~matilda_uci.matilda.search.SearchController`; without one it zeroes
out gracefully (in-distribution).

A ``wrapper``/``wrapper_factory`` can be injected to run without the real
Maia-3 runtime (used by the tests).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import torch

from .features import tensors_sf
from .model import N_CANDIDATES, TXTC
from .move_vocab import legal_mask, move_index
from .search import SearchController, scores_to_arrays

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MatildaPrediction:
    """One re-ranked move distribution for a position."""

    move_probs: dict[str, float]  # real-board uci -> prob, legal moves only
    win_prob: float  # Maia-3 expected score for the side to move, in [0,1]
    candidates: tuple[str, ...]  # the up-to-16 candidate UCIs the re-ranker saw
    prior_probs: dict[str, float]  # Maia-3's own distribution, for comparison
    engine_used: bool  # whether a search controller scored this position


class MatildaModel:
    """Loads the checkpoint family and predicts human move distributions.

    ``checkpoint`` is a TXTC state dict (``base_3k.pt`` — THE deploy model).
    Style personalization is optional: :meth:`load_style` overlays a style-token
    checkpoint (and, for new players, post-hoc embedding rows); pass ``pid`` to
    :meth:`predict` to condition on a player (0 = the trained generic fallback).
    """

    def __init__(
        self,
        checkpoint: str = "checkpoints/base_3k.pt",
        *,
        device: str = "cpu",
        maia3_model: str = "23m",
        wrapper: object | None = None,
        wrapper_factory: Callable[[], object] | None = None,
    ) -> None:
        self.checkpoint = checkpoint
        self.device = device
        self.maia3_model = maia3_model
        self._wrapper = wrapper
        self._wrapper_factory = wrapper_factory
        self._model: TXTC | None = None
        self._base_sd: dict | None = None
        self._style_loaded = False
        self._n_style_rows = 0

    # --- loading -----------------------------------------------------------------
    def _ensure_model(self) -> TXTC:
        if self._model is None:
            sd = torch.load(self.checkpoint, map_location="cpu")
            self._base_sd = {k: v for k, v in sd.items() if not k.startswith("sp")}
            model = TXTC()
            model.load_state_dict(sd, strict=True)
            model.to(self.device).eval()
            self._model = model
        return self._model

    def load_style(self, style_checkpoint: str, posthoc: str | None = None) -> int:
        """Overlay a style-token checkpoint (and optional post-hoc new players).

        Rebuilds the model with the overlay's full player table, exactly as the
        checkpoint README prescribes: base weights first (``sp*`` excluded),
        then the ``spe``/``spp`` overlay; post-hoc rows are written into
        ``spe.weight[n_old:]``. Returns the number of player rows available
        (valid ``pid`` range is ``0 .. rows-1``; 0 = generic fallback).
        """
        self._ensure_model()  # populates _base_sd
        assert self._base_sd is not None
        tok = torch.load(style_checkpoint, map_location="cpu")
        spe_overlay = tok["state_dict"]["spe.weight"]
        rows = spe_overlay.shape[0]

        ph = None
        if posthoc is not None:
            ph = torch.load(posthoc, map_location="cpu")
            n_old, spe_new = int(ph["n_old"]), ph["spe_new"]
            if n_old != rows:
                logger.warning(
                    "posthoc n_old=%d != style table rows=%d; embeddings were fit "
                    "in a different style space", n_old, rows,
                )
            rows = max(rows, n_old + spe_new.shape[0])

        model = TXTC(sdim=int(tok["sdim"]), n_players=rows - 1)
        model.load_state_dict(self._base_sd, strict=False)  # base, sans style
        # spp loads by key; spe is copied row-wise (the model's table may be
        # larger than the overlay's when post-hoc players extend it).
        spp_only = {k: v for k, v in tok["state_dict"].items() if not k.startswith("spe")}
        model.load_state_dict(spp_only, strict=False)
        with torch.no_grad():
            model.spe.weight[: spe_overlay.shape[0]] = spe_overlay
            if ph is not None:
                n_old = int(ph["n_old"])
                model.spe.weight[n_old : n_old + ph["spe_new"].shape[0]] = ph["spe_new"]
        model.to(self.device).eval()
        self._model = model
        self._style_loaded = True
        self._n_style_rows = rows
        return rows

    @property
    def style_rows(self) -> int:
        """Player rows available for ``pid`` (0 when no style overlay loaded)."""
        return self._n_style_rows

    def _ensure_wrapper(self) -> object:
        if self._wrapper is None:
            if self._wrapper_factory is not None:
                self._wrapper = self._wrapper_factory()
            else:
                from .maia3_wrapper import Maia3Wrapper

                self._wrapper = Maia3Wrapper(
                    model=self.maia3_model,
                    device=self.device if self.device != "mps" else "cpu",
                    top_k=N_CANDIDATES,
                    capture_hidden=True,
                    capture_importance=True,
                )
        return self._wrapper

    # --- inference ---------------------------------------------------------------
    def predict(
        self,
        board: "object",
        *,
        board_history: Sequence[str] | None = None,
        elo_self: int = 1500,
        elo_oppo: int = 1500,
        tc_base: float = 180.0,
        tc_inc: float = 0.0,
        controller: SearchController | None = None,
        pid: int | None = None,
    ) -> MatildaPrediction | None:
        """Re-ranked move distribution for ``board``; ``None`` if no legal moves.

        ``board_history`` is the preceding board FENs, oldest first (Maia-3
        consumes the trailing 8 — see the wrapper). ``pid`` requires a style
        overlay loaded via :meth:`load_style`.
        """
        import chess

        legal = [m.uci() for m in board.legal_moves]
        if not legal:
            return None
        white = board.turn == chess.WHITE

        wrapper = self._ensure_wrapper()
        r = wrapper.infer(
            board,
            elo_self=int(elo_self),
            elo_oppo=int(elo_oppo),
            board_history=list(board_history or []),
        )

        # --- candidate block, exactly as featurize_maia3_only.py builds it ---
        cand = [u for u, _ in r.top_moves[:N_CANDIDATES]]
        idx = np.full(N_CANDIDATES, -1, np.int64)
        lg = np.zeros(N_CANDIDATES, np.float32)
        valid = np.zeros(N_CANDIDATES, np.float32)
        padded: list[str] = [""] * N_CANDIDATES
        for j, u in enumerate(cand):
            vi = move_index(u, white)
            if vi is None:
                continue
            idx[j] = vi
            lg[j] = r.logit(u)
            valid[j] = 1.0
            padded[j] = u
        if valid.sum() == 0:
            logger.warning("no mappable candidates; falling back to Maia-3 prior")
            return MatildaPrediction(
                move_probs=dict(r.move_probs), win_prob=float(r.win_prob),
                candidates=(), prior_probs=dict(r.move_probs), engine_used=False,
            )

        hidden = np.asarray(r.hidden, np.float32).reshape(-1)
        imp = (
            np.asarray(r.importance, np.float32).reshape(8, 8)
            if r.importance is not None
            else np.zeros((8, 8), np.float32)
        )
        # Training shards stored these as float16; quantize identically.
        d = dict(
            maia_logits=np.asarray(r.logits, np.float16).reshape(-1)[:4352][None],
            maia_hidden=hidden.astype(np.float16)[None],
            maia_importance=imp.astype(np.float16)[None],
            legal_mask=legal_mask(board).astype(np.int8)[None],
            cand_idx=idx[None],
            cand_logit=lg[None],
            cand_valid=valid[None],
            tc_base=np.asarray([tc_base], np.float32),
            tc_inc=np.asarray([tc_inc], np.float32),
        )

        engine_used = False
        if controller is not None:
            try:
                scores = controller.score(board, [u for u in padded if u])
            except Exception:
                logger.exception("search controller failed; playing without it")
                scores = []
            sf_cp, sf_rank, sf_valid = scores_to_arrays(padded, scores)
            d["sf_cp"] = sf_cp[None]
            d["sf_rank"] = sf_rank[None]
            d["sf_valid"] = np.asarray([sf_valid], np.int8)
            engine_used = bool(sf_valid)

        T = tensors_sf(d, use_sf=True)
        model = self._ensure_model()
        T = {k: v.to(self.device) for k, v in T.items()}
        pid_t = None
        use_style = False
        if pid is not None:
            if self._style_loaded and 0 <= int(pid) < self._n_style_rows:
                pid_t = torch.tensor([int(pid)], device=self.device)
                use_style = True
            else:
                logger.warning(
                    "pid=%s ignored (style rows loaded: %d)", pid, self._n_style_rows
                )
        with torch.no_grad():
            lp = model(
                T["ml"], T["hid"], T["imp"], T["lm"], T["ci"], T["tok"], T["val"],
                T["tcf"], pid=pid_t, style=use_style,
            )[0].cpu()

        probs: dict[str, float] = {}
        for uci in legal:
            vi = move_index(uci, white)
            probs[uci] = float(lp[vi].exp()) if vi is not None else 0.0
        total = sum(probs.values())
        if total > 0:
            probs = {u: p / total for u, p in probs.items()}

        return MatildaPrediction(
            move_probs=probs,
            win_prob=float(r.win_prob),
            candidates=tuple(u for u in padded if u),
            prior_probs=dict(r.move_probs),
            engine_used=engine_used,
        )

    def close(self) -> None:
        if self._wrapper is not None:
            close = getattr(self._wrapper, "close", None)
            if callable(close):
                close()
            self._wrapper = None
