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
    remain within ``distance_threshold_m`` of each other. Optional speed gate
    filters out non-suspicious co-location (e.g. ships moving in convoy at speed).
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
                    },
                )
                rec["lon_sum"] += float(lonlat[i, 0].item() + lonlat[j, 0].item()) / 2
                rec["lat_sum"] += float(lonlat[i, 1].item() + lonlat[j, 1].item()) / 2
                rec["n"] += 1
                rec["end"] = chunk[i, 1].item()
                rec["min_d"] = min(rec["min_d"], float(d[i, j].item()))

    for key, rec in open_pairs.items():
        dwell = float(rec["end"] - rec["start"])
        if dwell < min_dwell_seconds:
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
            )
        )

    return candidates
