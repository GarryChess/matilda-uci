"""Feature builder for TXTC — faithful port of the paper's ``tensors_sf``.

Ported from ``pipeline/train/train_sf_smoke.py`` in the read-only paper repo
(github.com/GarryChess/matilda1-paper); parity is asserted by
``scripts/verify_checkpoint.py``. Turns the raw per-position arrays produced by
the Maia-3 wrapper (and, optionally, a search controller) into the tensor dict
:meth:`matilda_uci.matilda.model.TXTC.forward` consumes.

Candidate-token features, exactly the paper's list (Method + audits appendix):

* base block (``FD_BASE`` = 4): normalized Maia-3 logit, rank, gap-to-best,
  top-1 flag;
* engine block (``FD_SF`` = 5): scored flag, clamped cp, cp-loss-vs-best,
  engine rank, engine-top-1 flag — each multiplied by ``scored`` so rows
  without annotation (``sf_valid=0``) zero the block (the paper's
  graceful-degradation gating); ``use_sf=False`` zeroes it entirely.

Input dict ``d`` (numpy arrays; the featurizer/wrapper contract):

    maia_logits (N,4352)  maia_hidden (N,512)  maia_importance (N,8,8 or N,64)
    legal_mask (N,4352)   cand_idx/cand_logit/cand_valid (N,16)
    tc_base/tc_inc (N,)   sf_cp/sf_rank (N,16) + sf_valid (N,)   [optional]
    target (N,)                                                  [optional]

``cand_logit`` is the *log of the legal-masked softmax probability* of each
candidate (``Maia3Result.logit``), not the raw logit. ``sf_cp`` uses -32001 as
the "unscored" sentinel; ``sf_rank`` is 1-based (1 = engine best).
"""

from __future__ import annotations

import numpy as np
import torch

from .model import N_CANDIDATES, VOCAB_SIZE

# Sentinel for "this candidate was not scored by the engine".
SF_CP_UNSCORED = -32001


def tensors_sf(d: dict, use_sf: bool) -> dict[str, torch.Tensor]:
    """The paper's feature builder: raw arrays -> TXTC input tensors.

    Missing ``sf_*`` keys behave as unscored (the block zeroes out), so a pure
    Maia-3 caller can omit them entirely. ``target`` is included as ``tg`` only
    when present (training/eval data has it; live inference does not).
    """
    N = len(d["cand_valid"])
    cl = torch.tensor(np.asarray(d["cand_logit"]), dtype=torch.float32)
    val = torch.tensor(np.asarray(d["cand_valid"]), dtype=torch.float32)
    ci = torch.tensor(np.asarray(d["cand_idx"])).long().clamp(min=0)
    clm = cl.masked_fill(val == 0, -1e9)
    m3r = torch.zeros_like(cl)
    m3r.scatter_(
        1,
        clm.argsort(1, descending=True),
        torch.arange(N_CANDIDATES).float().expand(N, N_CANDIDATES),
    )
    m3g = (clm.max(1, keepdim=True).values - cl) * val
    base = [cl / 8, m3r / 16, m3g / 8, (m3r == 0).float() * val]

    cp = torch.tensor(
        np.asarray(d.get("sf_cp", np.full((N, N_CANDIDATES), SF_CP_UNSCORED))),
        dtype=torch.float32,
    )
    sf_valid = torch.tensor(
        np.asarray(d.get("sf_valid", np.zeros(N))), dtype=torch.float32
    )
    sf_rank = torch.tensor(
        np.asarray(d.get("sf_rank", np.zeros((N, N_CANDIDATES)))), dtype=torch.float32
    )
    scored = (cp > SF_CP_UNSCORED).float() * sf_valid.unsqueeze(1)
    cpz = torch.where(scored.bool(), cp, torch.zeros_like(cp))
    best = (cpz + (1 - scored) * -1e9).max(1, keepdim=True).values
    sf = [
        scored,
        (torch.clamp(cpz / 1000, -3, 3) / 3) * scored,
        (torch.clamp((best - cpz) / 500, 0, 4) / 4) * scored,
        (sf_rank / 16) * scored,
        (sf_rank == 1).float() * scored,
    ]
    if not use_sf:
        sf = [torch.zeros_like(s) for s in sf]
    tok = torch.stack(base + sf, -1)

    tcf = torch.stack(
        [
            torch.log1p(torch.tensor(np.asarray(d["tc_base"]), dtype=torch.float32)) / 7,
            torch.log1p(torch.tensor(np.asarray(d["tc_inc"]), dtype=torch.float32)) / 3,
        ],
        -1,
    )
    out = dict(
        ml=torch.tensor(np.asarray(d["maia_logits"]), dtype=torch.float32).reshape(
            N, VOCAB_SIZE
        ),
        hid=torch.tensor(np.asarray(d["maia_hidden"]), dtype=torch.float32),
        imp=torch.tensor(np.asarray(d["maia_importance"]), dtype=torch.float32).reshape(
            N, -1
        ),
        lm=torch.tensor(np.asarray(d["legal_mask"]), dtype=torch.float32),
        ci=ci,
        tok=tok,
        val=val,
        tcf=tcf,
    )
    if "target" in d:
        out["tg"] = torch.tensor(np.asarray(d["target"])).long()
    return out
