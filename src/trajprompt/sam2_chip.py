"""SAM 2 visual confirmation pipeline.

Given a candidate rendezvous, pull the colocated Sentinel-2 chip from
Microsoft Planetary Computer, run SAM 2 with a "ship" prompt, and verify
that a vessel is actually visible at the AIS coordinates.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import torch


@dataclass
class ChipResult:
    image_chip: torch.Tensor    # (3, H, W)
    sam2_mask: torch.Tensor     # (1, H, W)
    confidence: float


class Sam2ChipPipeline:
    """Stub pipeline.

    In production this would:
    - Use ``planetary_computer`` + STAC search to find Sentinel-2 imagery
      at (lon, lat, datetime).
    - Crop a 256x256 chip centered on the AIS coordinates.
    - Run SAM 2 with a "ship" text prompt or a point prompt at the AIS lat/lon.
    - Return the masked chip and a confidence score (mask area / chip area or
      the SAM 2 logit at the prompt).
    """

    def __init__(self, sam2_model_id: str = "facebook/sam2-hiera-large"):
        self.sam2_model_id = sam2_model_id

    def chip_for(self, lon: float, lat: float, ts: datetime) -> ChipResult:
        """Stub returning a synthetic chip + null mask. Wire in MS Planetary Computer for real use."""
        return ChipResult(
            image_chip=torch.zeros(3, 256, 256),
            sam2_mask=torch.zeros(1, 256, 256),
            confidence=0.0,
        )

    def confirm(self, lon: float, lat: float, ts: datetime, window_days: int = 3) -> ChipResult:
        """Try a +/- window_days window around ts to find a clear-sky chip."""
        for offset in range(window_days + 1):
            for sign in (1, -1):
                attempt_ts = ts + timedelta(days=sign * offset)
                chip = self.chip_for(lon, lat, attempt_ts)
                if chip.confidence > 0.5:
                    return chip
        return chip
