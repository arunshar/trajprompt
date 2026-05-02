"""Smoke tests for HF Space deployment of trajprompt."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SPACE_APP = REPO_ROOT / "space" / "mapbox_app.py"


def _load_app_module():
    spec = importlib.util.spec_from_file_location("trajprompt_space_app", SPACE_APP)
    module = importlib.util.module_from_spec(spec)
    sys.modules["trajprompt_space_app"] = module
    spec.loader.exec_module(module)
    return module


# -- package imports ---------------------------------------------------------


def test_top_level_imports():
    import trajprompt
    from trajprompt import find_rendezvous, RendezvousCandidate, TrajCLIPEncoder
    assert trajprompt.__version__


def test_tgard_imports():
    from trajprompt.tgard import find_rendezvous, haversine_pairwise


def test_traj_clip_imports():
    from trajprompt.traj_clip import TrajCLIPEncoder, trajclip_loss


def test_sam2_chip_imports():
    from trajprompt.sam2_chip import Sam2ChipPipeline, ChipResult


# -- end-to-end pipeline path -----------------------------------------------


def test_traj_clip_forward_e2e():
    from trajprompt.traj_clip import TrajCLIPEncoder
    enc = TrajCLIPEncoder(in_features=6, hidden=64, depth=1, embed_dim=32)
    traj = torch.randn(4, 16, 6)
    z = enc(traj)
    assert z.shape == (4, 32)
    assert torch.allclose(z.norm(dim=-1), torch.ones(4), atol=1e-5)


def test_tgard_pipeline_e2e():
    """A short synthetic AIS sample exercises the haversine + dwell pipeline."""
    from trajprompt.tgard import find_rendezvous
    rows = []
    for t in range(0, 1800, 60):
        rows.append([1234, t, 0.0, 0.0])
        rows.append([5678, t, 0.0001, 0.0])
    pts = torch.tensor(rows, dtype=torch.float64)
    out = find_rendezvous(pts, distance_threshold_m=500.0, min_dwell_seconds=600.0)
    assert isinstance(out, list)
    assert len(out) >= 1


def test_sam2_chip_pipeline_returns_correct_shape():
    """Stub pipeline returns a 3-channel chip + 1-channel mask."""
    from datetime import datetime
    from trajprompt.sam2_chip import Sam2ChipPipeline
    pipeline = Sam2ChipPipeline()
    chip = pipeline.chip_for(0.0, 0.0, datetime(2024, 1, 1))
    assert chip.image_chip.shape == (3, 256, 256)
    assert chip.sam2_mask.shape == (1, 256, 256)


# -- Gradio app smoke -------------------------------------------------------


def test_space_app_importable():
    module = _load_app_module()
    assert hasattr(module, "build_ui")
    assert hasattr(module, "search")


def test_space_ui_builds():
    gr = pytest.importorskip("gradio")
    module = _load_app_module()
    ui = module.build_ui()
    assert isinstance(ui, gr.Blocks)


def test_space_callback_returns_tuple():
    module = _load_app_module()
    out = module.search("ships near pipelines", 30)
    assert isinstance(out, tuple) and len(out) == 2


# -- requirements + readme --------------------------------------------------


def test_space_requirements_parseable():
    req = REPO_ROOT / "space" / "requirements.txt"
    assert req.exists()
    text = req.read_text().lower()
    assert "gradio" in text
    assert "torch" in text


def test_space_readme_has_hf_frontmatter():
    readme = REPO_ROOT / "space" / "README.md"
    assert readme.exists()
    body = readme.read_text()
    assert body.startswith("---\n")
    assert "sdk: gradio" in body
    assert "app_file:" in body
