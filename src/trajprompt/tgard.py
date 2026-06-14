"""TGARD: Trajectory Group Anomalous Rendezvous Detection (PyTorch port).

Original publication: Sharma et al., "Detecting Anomalous Rendezvous in Trajectories",
ACM SIGSPATIAL. The algorithm finds pairs (or larger groups) of vessels that
co-locate within a spatial threshold for a minimum dwell duration, optionally
filtered by behavioral conditions (low speed, course alignment).

This port preserves the original semantics but uses PyTorch tensors so the
output can flow into a downstream traj-CLIP scorer for prompt-based ranking.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class RendezvousCandidate:
    """A candidate rendezvous between two MMSIs."""

    mmsi_a: int
    mmsi_b: int
    start_ts: int             # epoch seconds
    end_ts: int
    centroid_lonlat: tuple    # (lon, lat) of the rendezvous mean
    dwell_seconds: float
    closest_distance_m: float
    max_speed_knots: float    # peak implied transit speed during the rendezvous


# 1 knot = 1 nautical mile/hour = 1852 m / 3600 s.
_KNOT_TO_M_PER_S = 1852.0 / 3600.0


def haversine_pairwise(lonlat_a: Tensor, lonlat_b: Tensor) -> Tensor:
    """Pairwise Haversine distance in meters. Inputs are (N, 2) and (M, 2)."""
    R = 6_371_000.0
    lat1, lat2 = torch.deg2rad(lonlat_a[:, 1])[:, None], torch.deg2rad(lonlat_b[:, 1])[None, :]
    dlat = lat2 - lat1
    dlon = torch.deg2rad(lonlat_b[:, 0])[None, :] - torch.deg2rad(lonlat_a[:, 0])[:, None]
    a = (dlat / 2).sin() ** 2 + lat1.cos() * lat2.cos() * (dlon / 2).sin() ** 2
    return 2.0 * R * torch.asin(a.sqrt().clamp(0, 1))


def find_rendezvous(
    points: Tensor,             # (N, 4): [mmsi, ts, lon, lat]
    *,
    distance_threshold_m: float = 500.0,
    min_dwell_seconds: float = 600.0,
    max_speed_knots: float | None = 3.0,
) -> list[RendezvousCandidate]:
    """Find anomalous rendezvous in an AIS sample.

    A rendezvous is a contiguous time interval during which two distinct MMSIs
    remain within ``distance_threshold_m`` of each other. The optional speed
    gate ``max_speed_knots`` filters out non-suspicious co-location (e.g. ships
    moving in convoy at speed): a candidate is kept only if the peak implied
    transit speed of the rendezvous centroid stays at or below the threshold.
    Pass ``None`` or ``0`` to disable the speed gate (backward-compatible).
    """
    if points.dim() != 2 or points.shape[-1] < 4:
        raise ValueError(f"points must be (N, >=4), got {tuple(points.shape)}")

    points = points[points[:, 1].argsort()]   # sort by timestamp
    candidates: list[RendezvousCandidate] = []

    # Group by timestamp bucket (default 1 minute) and check pairs.
    bucket_seconds = 60.0
    bucket_id = (points[:, 1] / bucket_seconds).long()
    unique_buckets = bucket_id.unique()

    open_pairs: dict[tuple, dict] = {}

    for b in unique_buckets:
        mask = bucket_id == b
        chunk = points[mask]
        mmsis = chunk[:, 0].long()
        if mmsis.numel() < 2:
            continue
        lonlat = chunk[:, 2:4]
        d = haversine_pairwise(lonlat, lonlat)
        # Lower triangle excluding diagonal.
        for i in range(mmsis.numel()):
            for j in range(i + 1, mmsis.numel()):
                if d[i, j].item() > distance_threshold_m:
                    continue
                if mmsis[i].item() == mmsis[j].item():
                    continue
                key = (int(mmsis[i].item()), int(mmsis[j].item()))
                key = (min(key), max(key))
                rec = open_pairs.setdefault(
                    key,
                    {
                        "start": chunk[i, 1].item(),
                        "lon_sum": 0.0,
                        "lat_sum": 0.0,
                        "n": 0,
                        "min_d": float("inf"),
                        "prev_lon": None,
                        "prev_lat": None,
                        "prev_ts": None,
                        "max_speed_mps": 0.0,
                    },
                )
                cen_lon = float(lonlat[i, 0].item() + lonlat[j, 0].item()) / 2
                cen_lat = float(lonlat[i, 1].item() + lonlat[j, 1].item()) / 2
                ts = chunk[i, 1].item()
                rec["lon_sum"] += cen_lon
                rec["lat_sum"] += cen_lat
                rec["n"] += 1
                rec["end"] = ts
                rec["min_d"] = min(rec["min_d"], float(d[i, j].item()))

                # Implied transit speed of the rendezvous centroid between
                # consecutive contributing time buckets, used by the speed gate.
                if rec["prev_ts"] is not None:
                    dt = float(ts - rec["prev_ts"])
                    if dt > 0:
                        prev = torch.tensor([[rec["prev_lon"], rec["prev_lat"]]], dtype=lonlat.dtype)
                        cur = torch.tensor([[cen_lon, cen_lat]], dtype=lonlat.dtype)
                        step_m = float(haversine_pairwise(prev, cur)[0, 0].item())
                        rec["max_speed_mps"] = max(rec["max_speed_mps"], step_m / dt)
                rec["prev_lon"] = cen_lon
                rec["prev_lat"] = cen_lat
                rec["prev_ts"] = ts

    # Convert the knot threshold to m/s once. None or 0 disables the gate.
    speed_gate_mps = (
        max_speed_knots * _KNOT_TO_M_PER_S
        if max_speed_knots
        else None
    )

    for key, rec in open_pairs.items():
        dwell = float(rec["end"] - rec["start"])
        if dwell < min_dwell_seconds:
            continue
        cand_max_speed_mps = rec["max_speed_mps"]
        # Speed gate: drop pairs whose centroid transited faster than the
        # threshold (they were moving together, not loitering / rendezvousing).
        if speed_gate_mps is not None and cand_max_speed_mps > speed_gate_mps:
            continue
        candidates.append(
            RendezvousCandidate(
                mmsi_a=key[0],
                mmsi_b=key[1],
                start_ts=int(rec["start"]),
                end_ts=int(rec["end"]),
                centroid_lonlat=(rec["lon_sum"] / max(rec["n"], 1), rec["lat_sum"] / max(rec["n"], 1)),
                dwell_seconds=dwell,
                closest_distance_m=rec["min_d"],
                max_speed_knots=cand_max_speed_mps / _KNOT_TO_M_PER_S,
            )
        )

    return candidates
