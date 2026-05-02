"""Tests for the TrajCLIPEncoder + contrastive loss."""
from __future__ import annotations

import torch

from trajprompt.traj_clip import TrajCLIPEncoder, trajclip_loss


def test_encoder_output_is_l2_normalized():
    enc = TrajCLIPEncoder(in_features=6, hidden=128, depth=2, embed_dim=64)
    traj = torch.randn(4, 32, 6)
    z = enc(traj)
    norms = z.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(4), atol=1e-5)


def test_loss_drops_when_pairs_are_aligned():
    """Losses on perfectly-aligned pairs are smaller than on shuffled pairs."""
    torch.manual_seed(0)
    z_traj = torch.randn(8, 64)
    z_traj = z_traj / z_traj.norm(dim=-1, keepdim=True)
    aligned_text = z_traj.clone()
    shuffled_text = z_traj[torch.randperm(8)]
    aligned_loss = trajclip_loss(z_traj, aligned_text)
    shuffled_loss = trajclip_loss(z_traj, shuffled_text)
    assert aligned_loss < shuffled_loss
