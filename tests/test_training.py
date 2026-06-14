"""Real traj-CLIP training: synthetic data, alignment, and checkpoint round-trip."""
from __future__ import annotations

import torch

from trajprompt.traj_clip import TextEncoder
from trajprompt.training import data as syn
from trajprompt.training.train import _evaluate, load_trajclip, train_synthetic


def test_synthetic_pairs_shapes_and_ranges():
    concepts = torch.arange(syn.N_CONCEPTS)
    traj, text = syn.make_batch(concepts, length=16, noise=0.0)
    assert traj.shape == (syn.N_CONCEPTS, 16, syn.FEATURE_DIM)
    assert text.shape == (syn.N_CONCEPTS, 4)
    assert int(text.max()) < syn.VOCAB_SIZE and int(text.min()) >= 1  # no <pad> in content
    # noise=0 must be deterministic
    traj2, _ = syn.make_batch(concepts, length=16, noise=0.0)
    assert torch.allclose(traj, traj2)


def test_eval_set_is_one_per_concept():
    traj, text = syn.eval_set()
    assert traj.shape[0] == syn.N_CONCEPTS
    # distinct concepts must yield distinct text token rows
    assert len({tuple(row.tolist()) for row in text}) == syn.N_CONCEPTS


def test_text_encoder_normalized():
    enc = TextEncoder(vocab_size=syn.VOCAB_SIZE, embed_dim=64, hidden=64)
    text = syn.concept_text_ids(torch.arange(8))
    z = enc(text)
    assert z.shape == (8, 64)
    assert torch.allclose(z.norm(dim=-1), torch.ones(8), atol=1e-5)


def test_training_aligns_encoder(tmp_path):
    report = train_synthetic(out_dir=str(tmp_path), steps=240, batch_size=64, lr=1e-3, seed=0)
    # loss must drop and retrieval must rise well above the 1/32 chance baseline
    assert report.loss_end < report.loss_start
    assert report.retrieval_acc_end > report.retrieval_acc_start
    assert report.retrieval_acc_start < 0.2          # starts near chance
    assert report.retrieval_acc_end > 0.5            # genuinely aligned


def test_checkpoint_roundtrip(tmp_path):
    report = train_synthetic(out_dir=str(tmp_path), steps=120, batch_size=64, lr=1e-3, seed=1)
    traj_enc, text_enc, ckpt = load_trajclip(report.checkpoint_path)
    assert ckpt["synthetic"] is True
    eval_traj, eval_text = syn.eval_set()
    _, acc = _evaluate(traj_enc, text_enc, eval_traj, eval_text)
    # reloaded encoders reproduce the end-of-training retrieval accuracy exactly
    assert abs(acc - report.retrieval_acc_end) < 1e-6
