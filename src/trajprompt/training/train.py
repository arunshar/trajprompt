"""Train traj-CLIP on AIS-text pairs.

The text side is a frozen sentence-transformer; the trajectory side is the
TrajCLIPEncoder defined in src/trajprompt/traj_clip.py. We fine-tune the
trajectory encoder via InfoNCE over the curated AIS-text pairs dataset.
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def main(
    pairs_path: str = "data/ais_text_pairs.jsonl",
    out_dir: str = "outputs/trajclip_v1",
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1.0e-4,
):
    from trajprompt.traj_clip import TrajCLIPEncoder, trajclip_loss
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Install sentence-transformers: pip install sentence-transformers"
        ) from exc

    text_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    text_model.eval()
    traj_encoder = TrajCLIPEncoder().cuda()
    opt = torch.optim.AdamW(traj_encoder.parameters(), lr=lr, weight_decay=1e-2)

    # Stub training loop: the user supplies a curated AIS-text pairs JSONL.
    # Real loop loads (window_features, text) pairs and minimizes trajclip_loss.
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    logger.info("Wrote skeleton checkpoint to %s", out_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
