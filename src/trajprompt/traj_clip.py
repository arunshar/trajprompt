"""Trajectory-CLIP: contrastive alignment of AIS trajectory features with text.

A small transformer encodes a fixed-window AIS trajectory (sequence of
(lon, lat, sog, cog, distance_to_coast, dwell_in_polygon)) into a 256-d
embedding aligned by InfoNCE with sentence embeddings of natural-language
descriptions.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class TrajCLIPEncoder(nn.Module):
    """Trajectory-side encoder.

    Input: (B, T, F) where T is the trajectory length and F is feature dim.
    Output: (B, embed_dim) L2-normalized embedding.
    """

    def __init__(self, in_features: int = 6, hidden: int = 256, depth: int = 4, embed_dim: int = 256):
        super().__init__()
        self.in_proj = nn.Linear(in_features, hidden)
        self.pos = nn.Parameter(torch.randn(1, 512, hidden) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=8, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.head = nn.Linear(hidden, embed_dim)

    def forward(self, traj: Tensor) -> Tensor:
        h = self.in_proj(traj) + self.pos[:, : traj.shape[1]]
        h = self.encoder(h)
        return F.normalize(self.head(h.mean(dim=1)), dim=-1)


def trajclip_loss(
    traj_emb: Tensor,
    text_emb: Tensor,
    temperature: float = 0.07,
) -> Tensor:
    """Symmetric InfoNCE loss; both embeddings already L2-normalized."""
    logits_t2x = (traj_emb @ text_emb.t()) / temperature
    logits_x2t = logits_t2x.t()
    targets = torch.arange(traj_emb.shape[0], device=traj_emb.device)
    return 0.5 * (F.cross_entropy(logits_t2x, targets) + F.cross_entropy(logits_x2t, targets))
