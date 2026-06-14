"""Synthetic concept-language pairs for traj-CLIP alignment.

Each of 32 concepts is a combination of four attributes that each map to a
distinct trajectory feature channel and to a distinct word in the description:

    speed     -> sog channel          -> {slow, fast}
    heading   -> cog channel + path   -> {northbound, eastbound, southbound, westbound}
    proximity -> distance_to_coast    -> {coastal, offshore}
    behavior  -> dwell + net motion   -> {loitering, transiting}

So a trajectory and its matching phrase share a latent concept, and InfoNCE has a
genuine alignment to learn. This is a self-contained stand-in for a curated
AIS-text dataset, not a substitute for one; it exists so the training loop is
real and measurable on CPU.
"""

from __future__ import annotations

import torch
from torch import Tensor

_SPEED = ["slow", "fast"]
_HEADING = ["northbound", "eastbound", "southbound", "westbound"]
_PROX = ["coastal", "offshore"]
_BEHAV = ["loitering", "transiting"]
_VOCAB = ["<pad>", *_SPEED, *_HEADING, *_PROX, *_BEHAV]
VOCAB = list(_VOCAB)
_WORD_ID = {w: i for i, w in enumerate(_VOCAB)}

# attribute -> physical value, indexed by the attribute's category
_SPEED_KN = torch.tensor([3.0, 16.0])
_HEADING_DEG = torch.tensor([0.0, 90.0, 180.0, 270.0])
_DIST_KM = torch.tensor([2.0, 40.0])
_DWELL = torch.tensor([0.8, 0.05])           # loitering high, transiting low
_NET_MOVE = torch.tensor([0.1, 1.0])         # loitering ~stationary, transiting steady

N_CONCEPTS = 2 * 4 * 2 * 2                    # 32
VOCAB_SIZE = len(_VOCAB)
FEATURE_DIM = 6


def _decompose(concepts: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    speed_i = concepts % 2
    heading_i = (concepts // 2) % 4
    prox_i = (concepts // 8) % 2
    behav_i = (concepts // 16) % 2
    return speed_i, heading_i, prox_i, behav_i


def concept_text_ids(concepts: Tensor) -> Tensor:
    """(B,) concept indices -> (B, 4) token ids [speed, heading, prox, behavior]."""
    speed_i, heading_i, prox_i, behav_i = _decompose(concepts)
    speed_ids = torch.tensor([_WORD_ID[w] for w in _SPEED])[speed_i]
    heading_ids = torch.tensor([_WORD_ID[w] for w in _HEADING])[heading_i]
    prox_ids = torch.tensor([_WORD_ID[w] for w in _PROX])[prox_i]
    behav_ids = torch.tensor([_WORD_ID[w] for w in _BEHAV])[behav_i]
    return torch.stack([speed_ids, heading_ids, prox_ids, behav_ids], dim=1)


def concept_trajectories(
    concepts: Tensor, *, length: int = 16, noise: float = 0.15,
    generator: torch.Generator | None = None,
) -> Tensor:
    """(B,) concept indices -> (B, T, 6) feature trajectories.

    Channels: (lon_rel, lat_rel, sog/20, cog/360, dist/50, dwell). With noise=0
    each concept yields a single clean trajectory (used for evaluation).
    """
    b = concepts.shape[0]
    speed_i, heading_i, prox_i, behav_i = _decompose(concepts)
    speed = _SPEED_KN[speed_i]
    heading = _HEADING_DEG[heading_i]
    dist = _DIST_KM[prox_i]
    dwell = _DWELL[behav_i]
    net = _NET_MOVE[behav_i]

    def noisy(scale: float, shape: tuple[int, ...]) -> Tensor:
        if noise == 0:
            return torch.zeros(shape)
        return torch.randn(*shape, generator=generator) * (scale * noise)

    steps = torch.arange(length, dtype=torch.float32)
    hr = torch.deg2rad(heading)
    # per-step displacement (relative path), proportional to speed and net motion
    dx = (net * torch.sin(hr) * speed / 16.0)[:, None] * steps[None, :]
    dy = (net * torch.cos(hr) * speed / 16.0)[:, None] * steps[None, :]
    lon_rel = dx + noisy(0.5, (b, length))
    lat_rel = dy + noisy(0.5, (b, length))
    sog = (speed[:, None] + noisy(2.0, (b, length))) / 20.0
    cog = (heading[:, None] + noisy(15.0, (b, length))) / 360.0
    distf = (dist[:, None] + noisy(3.0, (b, length))) / 50.0
    dwellf = (dwell[:, None] + noisy(0.05, (b, length))).clamp(0.0, 1.0)
    return torch.stack([lon_rel, lat_rel, sog, cog, distf, dwellf], dim=-1)


def make_batch(
    concepts: Tensor, *, length: int = 16, noise: float = 0.15,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """(B,) concepts -> ((B, T, 6) trajectories, (B, 4) text ids)."""
    return (
        concept_trajectories(concepts, length=length, noise=noise, generator=generator),
        concept_text_ids(concepts),
    )


def eval_set(*, length: int = 16) -> tuple[Tensor, Tensor]:
    """One clean trajectory + text per concept (N_CONCEPTS items). Noise-free."""
    concepts = torch.arange(N_CONCEPTS)
    return make_batch(concepts, length=length, noise=0.0)


__all__ = [
    "N_CONCEPTS",
    "VOCAB_SIZE",
    "FEATURE_DIM",
    "concept_text_ids",
    "concept_trajectories",
    "make_batch",
    "eval_set",
]
