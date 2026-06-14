# TrajPrompt

> Research scaffold for open-vocabulary maritime trajectory search. Work in preparation. Some components are implemented and unit-tested; others are stubs or not yet integrated. See the component status table below before relying on anything.

[![HF Space](https://img.shields.io/badge/%F0%9F%A4%97-HF%20Space%20(placeholder)-lightgrey)](https://huggingface.co/spaces/Arun0808/trajprompt)
![traj-CLIP](https://img.shields.io/badge/traj--CLIP-trained%20(synthetic)-yellowgreen)
![status](https://img.shields.io/badge/status-research%20scaffold-blue)

## What this is

Maritime domain awareness today still happens largely through SQL filters on AIS feeds. The goal of TrajPrompt is to explore open-vocabulary, natural-language search over ship traffic: type "ships drifting suspiciously near pipelines in the Gulf of Mexico last March" and surface candidate trajectories, with optional Sentinel-2 visual confirmation.

This repository is an early research scaffold, not a working product. The architecture is sketched end to end. The trajectory encoder, the contrastive training loop that aligns it to language, and the rendezvous detector are implemented and tested today; the rest is stubbed, not integrated, or a placeholder. The component status table below states exactly what is real.

## Component status

| Component | File | Status |
|---|---|---|
| Trajectory-CLIP Transformer encoder (InfoNCE loss) | `src/trajprompt/traj_clip.py` | Implemented + unit-tested. The model runs forward, produces L2-normalized embeddings, and the symmetric InfoNCE loss is correct. The contrastive loop now aligns it to language on a synthetic concept-language task (see the Training loop row). It has NOT been trained on real AIS-text pairs, and no checkpoint is shipped in the repo. |
| TGARD rendezvous detector (tensorized Haversine + dwell + speed gate) | `src/trajprompt/tgard.py` | Implemented + unit-tested on synthetic AIS tracks. |
| Training loop | `src/trajprompt/training/train.py` | Implemented + unit-tested. A real symmetric-InfoNCE loop (batching, optimizer steps, eval) with a learnable text encoder over a self-contained synthetic concept-language dataset (`src/trajprompt/training/data.py`). It aligns the trajectory encoder to language: top-1 trajectory→text retrieval over 32 held-out concepts rises from 6.25% (near the 1/32 chance baseline) to 100%, loss 3.99 → 0.01 in 320 CPU steps, and it writes a loadable checkpoint (`load_trajclip`). It trains on SYNTHETIC concept-language, NOT real AIS-text pairs; swap in a sentence-transformer and curated pairs for that. |
| SAM 2 + Sentinel-2 visual confirmation | `src/trajprompt/sam2_chip.py` | Stub. `chip_for(...)` returns an all-zeros chip and an all-zeros mask with confidence 0.0. No Microsoft Planetary Computer / STAC query and no SAM 2 inference are wired in. |
| Prithvi-2 geospatial foundation model | (not present) | Not integrated. There is no Prithvi-2 code in this repo yet. |
| HF Space demo | `space/mapbox_app.py` | Placeholder UI. The Gradio app builds, but the `search(...)` callback returns placeholder text and no map; it does not run the pipeline. |
| AIS-text pairs dataset | (not present) | Not released. No curated dataset is included in this repo. |

In short: the encoder, the contrastive training that aligns it (on synthetic concept-language), and the rendezvous detector are real and tested; everything that would make this an end-to-end "type a question, see confirmed vessels on a map" system on real data (real AIS-text pairs, SAM 2 / Sentinel-2, the Space) is still scaffold, stub, or placeholder.

## What actually runs today

These commands work on synthetic data with no downloads:

```bash
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -e ".[dev]"
pytest -q                                 # the Gradio Space tests skip without the [space] extra
```

You can also exercise the real components directly:

```python
import torch
from trajprompt.tgard import find_rendezvous
from trajprompt.traj_clip import TrajCLIPEncoder

# TGARD on a synthetic stationary pair -> one rendezvous candidate.
rows = []
for t in range(0, 1800, 60):
    rows.append([1234, t, 0.0, 0.0])
    rows.append([5678, t, 0.0001, 0.0])
pts = torch.tensor(rows, dtype=torch.float64)
print(find_rendezvous(pts, distance_threshold_m=500.0, min_dwell_seconds=600.0))

# traj-CLIP encoder forward pass (untrained weights).
enc = TrajCLIPEncoder(in_features=6)
print(enc(torch.randn(2, 32, 6)).shape)   # -> torch.Size([2, 256])
```

Train the encoder end to end on the self-contained synthetic concept-language task (no downloads, CPU; aligns trajectory and text, then writes a checkpoint):

```python
from trajprompt.training.train import train_synthetic, load_trajclip

report = train_synthetic(out_dir="outputs/trajclip_v1", steps=320, seed=0)
print(report.retrieval_acc_start, "->", report.retrieval_acc_end)   # ~0.06 -> 1.0 over 32 concepts
traj_enc, text_enc, ckpt = load_trajclip(report.checkpoint_path)    # reload the aligned encoders
```

## Tests

```bash
pytest -q
```

Test breakdown (19 total):
- `tests/test_tgard.py` (5): Haversine self-distance and known-distance checks, no-match when far apart, rendezvous detection on a synthetic stationary dwell, and the speed gate filtering a fast convoy.
- `tests/test_traj_clip.py` (2): encoder output is L2-normalized; InfoNCE loss is lower for aligned than for shuffled pairs.
- `tests/test_smoke.py` (12): package/module imports, traj-CLIP forward shape, TGARD pipeline on synthetic data, the SAM 2 chip stub returns the right shapes, and Space-app smoke checks (the Gradio UI build, callback shape, requirements, and HF README frontmatter). The Gradio-dependent checks require the `[space]` extra and skip or error without it.

The TGARD and traj-CLIP tests need only PyTorch. The Space tests additionally need `pip install -e ".[space]"` (gradio).

## Not yet runnable as documented

The following are part of the intended design but do NOT work yet. They are listed so nobody mistakes the scaffold for a finished pipeline:

```bash
# Pulls Danish Maritime Authority AIS dumps. The downloader uses macOS `date -j`
# flags and is not portable to GNU/Linux date as written.
bash scripts/download_ais_dma.sh

# There is no Hydra config / +experiment plumbing, no data loader, and no
# optimization loop. This builds an untrained encoder and returns; it does NOT
# train traj-CLIP and does NOT produce a usable checkpoint.
python -m trajprompt.training.train
```

## Repository layout

```
trajprompt/
├── src/trajprompt/
│   ├── tgard.py                # TGARD rendezvous detection (tensorized) -- real, tested
│   ├── traj_clip.py            # trajectory-CLIP encoder + InfoNCE loss   -- real, tested
│   ├── sam2_chip.py            # Sentinel-2 chip + SAM 2 confirmation     -- STUB (returns zeros)
│   └── training/train.py       # training entry point                    -- NO-OP scaffold
├── space/mapbox_app.py         # Gradio Space                            -- placeholder callback
├── notebooks/dark_vessel_demo.ipynb
├── tests/                      # TGARD + traj-CLIP + smoke tests
└── paper/main.tex              # draft, in preparation
```

## Live demo

[HF Space](https://huggingface.co/spaces/Arun0808/trajprompt) is a placeholder: the UI loads, but the search callback returns placeholder text rather than running the pipeline.

## License

Apache 2.0.
