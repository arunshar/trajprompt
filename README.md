# TrajPrompt

> Open-vocabulary maritime intelligence. Type a question, surface vessel trajectories on a live map with Sentinel-2 visual confirmation. SAM 2 + Prithvi-2 + TGARD.

[![HF Space](https://img.shields.io/badge/%F0%9F%A4%97-HF%20Space-yellow)](https://huggingface.co/spaces/arun08sharma/trajprompt)
[![HF Model](https://img.shields.io/badge/%F0%9F%A4%97-traj--CLIP-blue)](https://huggingface.co/arun08sharma/trajprompt-clip-ais)
[![HF Dataset](https://img.shields.io/badge/%F0%9F%A4%97-AIS%20text%20pairs-orange)](https://huggingface.co/datasets/arun08sharma/ais-text-pairs)

Maritime domain awareness today still happens through SQL filters on AIS feeds. TrajPrompt brings open-vocabulary, natural-language search to the world's ship traffic. Type "ships drifting suspiciously near pipelines in the Gulf of Mexico last March" and the system surfaces candidate trajectories on a live Mapbox map with SAM 2 visual confirmation chips from Sentinel-2.

The core contribution is a trajectory-CLIP encoder that aligns AIS trajectory features with natural-language descriptions, combined with the published TGARD rendezvous detection algorithm and a Sentinel-2 vision pipeline.

## Highlights

- Trajectory-CLIP encoder: aligns AIS feature vectors (speed-over-ground, course-over-ground, distance-to-coast, dwell time) with text via contrastive InfoNCE.
- TGARD trajectory rendezvous detection (Sharma et al., SIGSPATIAL): identifies suspicious meet-ups on AIS data.
- SAM 2 visual confirmation: each candidate trajectory is verified by segmenting the actual vessel in a Sentinel-2 chip pulled from Microsoft Planetary Computer.
- Curated AIS-text pairs dataset for the trajectory-CLIP fine-tune, released as a NeurIPS Datasets & Benchmarks 2026 submission.

## Quickstart

```bash
git clone https://github.com/arunshar/trajprompt
cd trajprompt
pip install -e .
bash scripts/download_ais_dma.sh
python -m trajprompt.training.train +experiment=trajclip_ais
```

## Try the live demo

[HF Space](https://huggingface.co/spaces/arun08sharma/trajprompt) — Mapbox dark theme, free-text prompt, output is annotated trajectory traces + Sentinel-2 chips with SAM 2 masks.

## Repository layout

```
trajprompt/
├── src/trajprompt/
│   ├── tgard.py                # TGARD rendezvous detection (PyTorch port)
│   ├── traj_clip.py            # trajectory-CLIP encoder + contrastive trainer
│   ├── sam2_chip.py            # Sentinel-2 chip puller + SAM 2 inference
│   ├── data/{ais.py, sentinel2.py}
│   └── training/train.py
├── space/mapbox_app.py
├── notebooks/dark_vessel_demo.ipynb
├── tests/                      # TGARD + traj-CLIP + chip puller tests
└── paper/main.tex              # CVPR EarthVision 2027 + NeurIPS D&B 2026
```

## Reproduce the dark-vessel demo

```bash
jupyter lab notebooks/dark_vessel_demo.ipynb
```

## License

Apache 2.0.
