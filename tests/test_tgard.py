"""Tests for TGARD trajectory rendezvous detection."""
from __future__ import annotations

import torch

from trajprompt.tgard import find_rendezvous, haversine_pairwise


def test_haversine_self_distance_zero():
    pts = torch.tensor([[0.0, 0.0], [10.0, 10.0]])
    d = haversine_pairwise(pts, pts)
    assert d.diagonal().abs().max() < 1.0


def test_haversine_known_distance():
    """Approx ~111 km per degree of latitude at the equator."""
    a = torch.tensor([[0.0, 0.0]])
    b = torch.tensor([[0.0, 1.0]])
    d = haversine_pairwise(a, b).item()
    assert 110_000 < d < 112_000


def test_rendezvous_no_match_when_far_apart():
    """Two ships 1 km apart for 1 hour -> not within 500 m -> no rendezvous."""
    rows = []
    for t in range(3600):
        rows.append([1234, t, 0.0, 0.0])
        rows.append([5678, t, 0.01, 0.0])  # ~1.1 km away
    pts = torch.tensor(rows, dtype=torch.float64)
    out = find_rendezvous(pts, distance_threshold_m=500.0, min_dwell_seconds=600.0)
    assert out == []


def test_rendezvous_detects_close_dwell():
    """Two ships sitting within 100 m for 30 min -> one detected rendezvous."""
    rows = []
    for t in range(0, 1800, 60):  # 30 min, 1-minute samples
        rows.append([1234, t, 0.0, 0.0])
        rows.append([5678, t, 0.0001, 0.0])  # ~11 m east
    pts = torch.tensor(rows, dtype=torch.float64)
    out = find_rendezvous(pts, distance_threshold_m=500.0, min_dwell_seconds=600.0)
    assert len(out) == 1
    cand = out[0]
    assert {cand.mmsi_a, cand.mmsi_b} == {1234, 5678}
    assert cand.dwell_seconds >= 600.0
    # Stationary pair: centroid barely moves, so implied speed is well under 3 kn.
    assert cand.max_speed_knots < 3.0


def test_speed_gate_filters_fast_convoy():
    """Two ships moving together at speed stay within 500 m but transit fast.

    They co-locate for 30 minutes (a long dwell) yet steam ~0.01 deg of
    longitude per minute (roughly 20 kn at the equator). The default 3-kn
    speed gate should reject them; disabling the gate should keep them.
    """
    rows = []
    for k, t in enumerate(range(0, 1800, 60)):  # 30 min, 1-minute samples
        lon = 0.01 * k  # convoy advances east each minute
        rows.append([1234, t, lon, 0.0])
        rows.append([5678, t, lon + 0.0001, 0.0])  # ~11 m apart, moving together
    pts = torch.tensor(rows, dtype=torch.float64)

    gated = find_rendezvous(
        pts, distance_threshold_m=500.0, min_dwell_seconds=600.0, max_speed_knots=3.0
    )
    assert gated == []  # fast convoy filtered out

    ungated = find_rendezvous(
        pts, distance_threshold_m=500.0, min_dwell_seconds=600.0, max_speed_knots=None
    )
    assert len(ungated) == 1  # gate disabled -> candidate retained
    assert ungated[0].max_speed_knots > 3.0
