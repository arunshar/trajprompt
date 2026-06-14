"""Training entry point for traj-CLIP.

Runs a real symmetric-InfoNCE alignment between the `TrajCLIPEncoder` and a
learnable `TextEncoder` over the self-contained synthetic concept-language
dataset (`trajprompt.training.data`): a genuine optimization loop with batching
that aligns the trajectory encoder to language and writes a loadable checkpoint.

This trains on SYNTHETIC concept-language, not real AIS-text pairs. Real-data
alignment is a drop-in: supply curated `(window_features, description)` pairs and
swap `TextEncoder` for a frozen sentence-transformer (same normalized-output
contract); the loop below is unchanged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch

from trajprompt.traj_clip import TextEncoder, TrajCLIPEncoder, trajclip_loss
from trajprompt.training import data as syn

logger = logging.getLogger(__name__)


@dataclass
class TrainReport:
    steps: int
    n_concepts: int
    loss_start: float
    loss_end: float
    retrieval_acc_start: float
    retrieval_acc_end: float
    checkpoint_path: str


@torch.no_grad()
def _evaluate(
    traj_enc: TrajCLIPEncoder, text_enc: TextEncoder, traj: torch.Tensor, text: torch.Tensor
) -> tuple[float, float]:
    """Loss + top-1 trajectory->text retrieval accuracy on a clean per-concept set."""
    was_training = traj_enc.training
    traj_enc.eval()
    text_enc.eval()
    zt = traj_enc(traj)
    zx = text_enc(text)
    loss = float(trajclip_loss(zt, zx).item())
    top1 = (zt @ zx.t()).argmax(dim=1)
    acc = float((top1 == torch.arange(traj.shape[0], device=traj.device)).float().mean().item())
    if was_training:
        traj_enc.train()
        text_enc.train()
    return loss, acc


def train_synthetic(
    out_dir: str = "outputs/trajclip_v1",
    *,
    steps: int = 320,
    batch_size: int = 64,
    lr: float = 1.0e-3,
    seed: int = 0,
    length: int = 16,
    noise: float = 0.15,
    embed_dim: int = 128,
    hidden: int = 128,
    depth: int = 2,
    device: str | None = None,
) -> TrainReport:
    """Align the trajectory and text encoders by InfoNCE on synthetic pairs."""
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    traj_enc = TrajCLIPEncoder(
        in_features=syn.FEATURE_DIM, hidden=hidden, depth=depth, embed_dim=embed_dim
    ).to(device)
    text_enc = TextEncoder(vocab_size=syn.VOCAB_SIZE, embed_dim=embed_dim, hidden=hidden).to(device)
    opt = torch.optim.AdamW(
        list(traj_enc.parameters()) + list(text_enc.parameters()), lr=lr, weight_decay=1e-2
    )
    gen = torch.Generator().manual_seed(seed + 1)

    eval_traj, eval_text = syn.eval_set(length=length)
    eval_traj, eval_text = eval_traj.to(device), eval_text.to(device)
    loss_start, acc_start = _evaluate(traj_enc, text_enc, eval_traj, eval_text)

    for _step in range(steps):
        concepts = torch.randint(0, syn.N_CONCEPTS, (batch_size,), generator=gen)
        traj, text = syn.make_batch(concepts, length=length, noise=noise, generator=gen)
        traj, text = traj.to(device), text.to(device)
        loss = trajclip_loss(traj_enc(traj), text_enc(text))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    loss_end, acc_end = _evaluate(traj_enc, text_enc, eval_traj, eval_text)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_path = out / "trajclip.pt"
    torch.save(
        {
            "traj_encoder": traj_enc.state_dict(),
            "text_encoder": text_enc.state_dict(),
            "config": {
                "in_features": syn.FEATURE_DIM, "hidden": hidden, "depth": depth,
                "embed_dim": embed_dim, "vocab_size": syn.VOCAB_SIZE, "length": length,
            },
            "vocab": syn.VOCAB,
            "synthetic": True,
        },
        ckpt_path,
    )
    logger.info(
        "trajclip aligned on synthetic concept-language: retrieval %.3f -> %.3f, "
        "loss %.3f -> %.3f, checkpoint=%s",
        acc_start, acc_end, loss_start, loss_end, ckpt_path,
    )
    return TrainReport(steps, syn.N_CONCEPTS, loss_start, loss_end, acc_start, acc_end, str(ckpt_path))


def load_trajclip(
    path: str, map_location: str = "cpu"
) -> tuple[TrajCLIPEncoder, TextEncoder, dict]:
    """Rebuild both encoders from a checkpoint written by `train_synthetic`."""
    ckpt = torch.load(path, map_location=map_location)
    cfg = ckpt["config"]
    traj_enc = TrajCLIPEncoder(
        in_features=cfg["in_features"], hidden=cfg["hidden"],
        depth=cfg["depth"], embed_dim=cfg["embed_dim"],
    )
    text_enc = TextEncoder(vocab_size=cfg["vocab_size"], embed_dim=cfg["embed_dim"], hidden=cfg["hidden"])
    traj_enc.load_state_dict(ckpt["traj_encoder"])
    text_enc.load_state_dict(ckpt["text_encoder"])
    traj_enc.eval()
    text_enc.eval()
    return traj_enc, text_enc, ckpt


def main(
    pairs_path: str = "data/ais_text_pairs.jsonl",
    out_dir: str = "outputs/trajclip_v1",
    epochs: int = 40,
    batch_size: int = 64,
    lr: float = 1.0e-3,
) -> TrainReport:
    # No real AIS-text dataset ships with the repo, so train on the self-contained
    # synthetic concept-language task: a genuine InfoNCE alignment that writes a
    # loadable checkpoint. For real AIS-text alignment, supply `pairs_path` and a
    # sentence-transformer text model (see the README); the loop is the same.
    report = train_synthetic(out_dir=out_dir, steps=epochs * 8, batch_size=batch_size, lr=lr)
    logger.warning(
        "Trained on SYNTHETIC concept-language pairs (not real AIS-text). "
        "Trajectory->text retrieval over %d concepts: %.1f%% -> %.1f%%. Checkpoint: %s",
        report.n_concepts, 100 * report.retrieval_acc_start,
        100 * report.retrieval_acc_end, report.checkpoint_path,
    )
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
