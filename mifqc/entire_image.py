# mifqc/entire_image.py
from __future__ import annotations
import numpy as np
from .image_base import ImageBase
from .qc_metrics import tissue_mask
from dataclasses import dataclass

@dataclass
class EntireImage(ImageBase):
    """Full-resolution multi-channel image."""

    # ---------- Whole-image QC extras ----------
    def tissue_fraction(self, nuclear_channel: str = "DAPI") -> float:
        idx = self.channel_names.index(nuclear_channel)
        mask = tissue_mask(self.pixels[idx])
        return float(mask.mean())
