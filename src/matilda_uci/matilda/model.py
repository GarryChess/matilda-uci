"""The Matilda set-transformer (``TXTC``) — faithful port of the paper pipeline.

``TXTC`` ("transformer, time-control") re-ranks frozen **Maia-3 (23M)**: it never
sees the board directly. It consumes Maia-3's outputs plus a small feature block
and emits one *residual delta* ("nudge") per candidate move, added back into
Maia-3's full-vocabulary policy logits.

Provenance
----------
Ported line-for-line from the read-only paper repo
(github.com/GarryChess/matilda1-paper): ``pipeline/train/train_base_tc.py``
(class ``TXTC``) with the candidate projection widened for the engine-feature
block exactly as ``pipeline/train/train_sf_smoke.py`` does for the shipped
checkpoints. Numerical parity with the paper implementation is asserted by
``scripts/verify_checkpoint.py`` (same weights + same inputs into both
forwards -> identical outputs), alongside a ``strict=True`` load of the real
``base_3k.pt`` (1,672,449 params).

Two deliberate deviations from the paper class's *defaults* (not its math),
both to make the no-argument constructor safe for the shipped checkpoints:

* ``heads=8`` (paper class default is 4, but every shipped checkpoint was
  trained with ``TXTC(heads=8)``; head count changes attention math without
  changing parameter shapes, so a wrong default would load cleanly and then
  silently compute garbage);
* ``fd=9`` (= ``FD_BASE + FD_SF``): the paper class default is the 4-feature
  TC-era block, and the training code widens ``tp`` to 4+5+32 -> 128 by hand.
  The shipped ``base_3k.pt`` / ``base_hi.pt`` have the widened ``tp`` baked in,
  so our default builds it directly.

State-dict layout (exact; load-verified against ``base_3k.pt``):

    le.0  Linear(4352->256)  le.3  Linear(256->128)   # + GELU/Dropout between
    ae.0  Linear(64->64)     he.0  Linear(512->128)
    ct    Linear(322->128)   # cat[le(128), ae(64), he(128), tc(2)]
    me    Embedding(4352,32) # learned move-index embedding
    tp    Linear(41->128)    # cat[9 candidate features, 32 move emb]
    enc   TransformerEncoder(d=128, heads=8, ff=256, L=2, gelu)
    dl    Linear(128->1)     # residual head; zero-init => exactly Maia-3
    spe   Embedding(n_players+1, 32)  spp  Linear(32->128)  # style (optional)

Feature contract (inputs to :meth:`TXTC.forward`)
-------------------------------------------------
``N`` = batch. Built by :mod:`matilda_uci.matilda.features` from the Maia-3
wrapper outputs:

    ml   (N, 4352)      Maia-3 policy logits, side-to-move frame
    hid  (N, 512)       Maia-3 pooled pre-logits hidden state (L2-normalized
                        inside ``forward``; pass it raw)
    imp  (N, 64)        Maia-3 per-square attention (8x8 flattened)
    lm   (N, 4352)      legal-move mask as float 0/1
    ci   (N, 16) long   vocab indices of the top-16 candidates (invalid -> 0)
    tok  (N, 16, 9)     per-candidate features (see ``features.tensors_sf``)
    val  (N, 16)        candidate valid mask as float 0/1
    tcf  (N, 2)         (log1p(base_s)/7, log1p(inc_s)/3)
    pid  (N,) long      optional player id (row in ``spe``; 0 = generic)
    style bool          whether to add the style vector to the context token
"""

from __future__ import annotations

import torch
import torch.nn as nn

# --- fixed architecture constants ---------------------------------------------
VOCAB_SIZE = 4352  # Maia-3 move vocabulary: 64*64 from->to + 256 promotions
HIDDEN_DIM = 512  # Maia-3 pooled pre-logits hidden state
IMPORTANCE_DIM = 64  # Maia-3 per-square attention map, 8x8 flattened
N_CANDIDATES = 16  # top-k candidate moves fed as tokens
D_MODEL = 128  # set-transformer working width
TC_DIM = 2  # (log1p(base)/7, log1p(inc)/3)
FD_BASE = 4  # per-candidate prior stats: logit, rank, gap-to-best, top-1 flag
FD_SF = 5  # per-candidate engine block: scored, cp, cp-loss, rank, engine-top-1
FD_CAND = FD_BASE + FD_SF  # 9 raw features per candidate in the shipped models
MOVE_EMB_DIM = 32  # learned per-vocab-move embedding
STYLE_DIM = 32  # player style embedding width (sdim)


class TXTC(nn.Module):
    """The paper's set-transformer re-ranker (Method, "Set-transformer base with
    full-vocabulary re-ranking"). Mapping to the paper:

    - ``le``/``ae``/``he`` encoders + ``ct`` -> the CONTEXT TOKEN z0 (Maia-3
      logits, attention, hidden state, and the time-control pair)
    - ``me`` + ``tp`` -> CANDIDATE TOKENS z1..zk (per-move features + learned
      move-index embedding)
    - ``enc`` (no positional encoding) -> permutation-invariant attention over
      {z0..zk}
    - ``dl`` (ZERO-INITIALIZED) -> per-candidate residual delta_i; zero init
      gives the paper's untrained-model == Maia-3 property
    - scatter_add + softmax -> "residual re-ranking": deltas are added back into
      the FULL 4352-d Maia-3 logits, normalized over all legal moves
    - ``spe``/``spp`` (zero-init projection) -> the 32-d player-style token
      added to z0 (trained with the base frozen)
    """

    def __init__(
        self,
        L: int = 2,
        H: int = D_MODEL,
        heads: int = 8,  # deviation: paper class default is 4; checkpoints use 8
        ff: int = 256,
        drop: float = 0.1,
        emb: int = MOVE_EMB_DIM,
        le: int = 128,
        sdim: int = STYLE_DIM,
        n_players: int = 1,
        fd: int = FD_CAND,  # deviation: paper default is FD_BASE; checkpoints use 9
    ) -> None:
        super().__init__()
        self.le = nn.Sequential(
            nn.Linear(VOCAB_SIZE, 256), nn.GELU(), nn.Dropout(drop),
            nn.Linear(256, le), nn.GELU(),
        )
        self.ae = nn.Sequential(nn.Linear(IMPORTANCE_DIM, 64), nn.GELU())
        self.he = nn.Sequential(nn.Linear(HIDDEN_DIM, 128), nn.GELU(), nn.Dropout(drop))
        self.ct = nn.Linear(le + 64 + 128 + TC_DIM, H)  # +2: the TC pair
        self.me = nn.Embedding(VOCAB_SIZE, emb)
        nn.init.normal_(self.me.weight, std=0.02)
        self.tp = nn.Linear(fd + emb, H)
        self.enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                H, heads, ff, dropout=drop, batch_first=True, activation="gelu"
            ),
            L,
            enable_nested_tensor=False,
        )
        self.dl = nn.Linear(H, 1)
        nn.init.zeros_(self.dl.weight)
        nn.init.zeros_(self.dl.bias)
        self.spe = nn.Embedding(n_players + 1, sdim)
        nn.init.normal_(self.spe.weight, std=0.02)
        self.spp = nn.Linear(sdim, H)
        nn.init.zeros_(self.spp.weight)
        nn.init.zeros_(self.spp.bias)

    def forward(
        self,
        mlb: torch.Tensor,
        hb: torch.Tensor,
        ib: torch.Tensor,
        lmb: torch.Tensor,
        cib: torch.Tensor,
        tokb: torch.Tensor,
        vb: torch.Tensor,
        tcf: torch.Tensor,
        pid: torch.Tensor | None = None,
        style: bool = True,
    ) -> torch.Tensor:
        """Full-vocabulary log-probabilities ``(N, 4352)`` over legal moves."""
        hb = hb / (hb.norm(dim=1, keepdim=True) + 1e-6)
        ct = self.ct(torch.cat([self.le(mlb), self.ae(ib), self.he(hb), tcf], -1))
        if style and pid is not None:
            ct = ct + self.spp(self.spe(pid))
        ct = ct.unsqueeze(1)
        cand = self.tp(torch.cat([tokb, self.me(cib) * vb.unsqueeze(-1)], -1))
        pad = torch.cat(
            [torch.zeros(len(mlb), 1, device=mlb.device), (vb == 0).float()], 1
        ).bool()
        o = self.enc(torch.cat([ct, cand], 1), src_key_padding_mask=pad)
        delta = self.dl(o[:, 1:]).squeeze(-1) * vb
        resid = torch.zeros_like(mlb)
        resid.scatter_add_(1, cib, delta)
        return torch.log_softmax(mlb + resid + torch.clamp(torch.log(lmb), min=-1e9), -1)

    def reset_nudge_head(self) -> None:
        """Zero the residual head: the model becomes *exactly* Maia-3."""
        nn.init.zeros_(self.dl.weight)
        nn.init.zeros_(self.dl.bias)


def maia3_reference_logprob(ml: torch.Tensor, lm: torch.Tensor) -> torch.Tensor:
    """Maia-3's own legal-masked log-softmax — the zero-delta reference.

    Exactly the ``lp3`` baseline from the paper's eval loops:
    ``log_softmax(ml + clamp(log(lm), min=-1e9))``.
    """
    return torch.log_softmax(ml + torch.clamp(torch.log(lm), min=-1e9), -1)
