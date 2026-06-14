"""Training entry point for traj-CLIP -- NO-OP SCAFFOLD, NOT A REAL TRAINER.

The intended design: freeze a sentence-transformer text encoder, and fine-tune
the trajectory-side TrajCLIPEncoder (src/trajprompt/traj_clip.py) via InfoNCE
over a curated AIS-text pairs dataset.

What this file actually does today: it builds an UNTRAINED encoder, creates the
output directory, and returns. There is no data loader, no batching loop, and no
optimization step, so:
  - the trajectory encoder is NOT aligned to language, and
  - NO usable checkpoint is written.

Do not treat a run of this module as having trained anything. The real loop
(load (window_features, text) pairs, embed text, minimize trajclip_loss, step
the optimizer, save weights) is not implemented yet.
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    text_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    text_model.eval()
    traj_encoder = TrajCLIPEncoder().to(device)
    # An optimizer is constructed to document intent, but nothing steps it:
    # there is no data loader and no training loop here yet.
    _opt = torch.optim.AdamW(traj_encoder.parameters(), lr=lr, weight_decay=1e-2)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    logger.warning(
        "trajprompt.training.train is a NO-OP scaffold: the encoder was NOT "
        "trained and NO checkpoint was written to %s. Implement the data loader "
        "and InfoNCE loop before using this for real training.",
        out_dir,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
