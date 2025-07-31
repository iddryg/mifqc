# mifqc/image_base.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from tifffile import TiffFile,imread
from typing import Sequence, Tuple, List, Optional, Union
import warnings
import dask.array as da
import dask_image.imread as dair
import zarr
from ome_types import from_tiff
from dataclasses import dataclass, field
from .qc_metrics import basic_stats, gini_index, moran_i

@dataclass
class ImageBase:
    """Abstract base for EntireImage and TiledImage."""
    pixels: np.ndarray                             # (C, H, W)
    channel_names: List[str]                       # length == C
    name: str = "unnamed"
    stats_: pd.DataFrame = field(init=False)

    # ---------- Constructors ----------
    @classmethod
    def from_tiff(cls, path: Union[str, Path], axes: Optional[str] = None,
              channel_names: Optional[List[str]] = None):
        with TiffFile(path) as tf:
            ser = tf.series[0]          # handle multi-series separately if needed
            axes_in = ser.axes if axes is None else axes
            arr = ser.asarray()         # honours memory-mapping or zarr if selected

            # Extract channel names INSIDE the context manager
            if channel_names is None:
                channel_names = _extract_channel_names(tf)

        if axes_in != "CYX":
            arr, axes_out = _canonicalize_axes(arr, axes_in, want="CYX")
            warnings.warn(f"Reordered axes {axes_in} → {axes_out}")
        return cls(arr, channel_names, name=Path(path).stem)

    # ---------- lazy TIFF/Zarr/Dask loading helpers ----------
    @classmethod
    def from_dask(cls, darr: da.Array, channel_names: Optional[List[str]] = None, name="unnamed"):
        if darr.ndim == 2:
            darr = darr[None, ...]
        if channel_names is None:
            channel_names = [f"ch{i}" for i in range(darr.shape[0])]
        return cls(darr, channel_names, name=name)

    @classmethod
    def from_zarr(cls, url: str, component: Optional[str] = None,
                  channel_names: Optional[List[str]] = None):
        darr = da.from_zarr(url, component=component)
        return cls.from_dask(darr, channel_names, name=Path(url).stem)

    @classmethod
    def from_glob(cls, pattern: str, channel_names: Optional[List[str]] = None):
        """Load N separate TIFFs (or PNGs) lazily as C×H×W via dask-image."""
        darr = dair.imread(pattern)      # shape = (N, H, W)
        return cls.from_dask(darr, channel_names, name=Path(pattern).stem)


    # ---------- Pixel-wise statistics ----------
    def per_channel_stats(self) -> pd.DataFrame:
        rows: List[dict] = []
        for c, ch_name in enumerate(self.channel_names):
            band = self.pixels[c]
            d = basic_stats(band)
            d["gini"] = gini_index(band)
            d["moran_I"] = moran_i(band)
            d["channel"] = ch_name
            rows.append(d)
        self.stats_ = pd.DataFrame(rows).set_index("channel")
        return self.stats_

    # ---------- Export ----------
    def to_csv(self, out_path: Union[str, Path]) -> None:
        if not hasattr(self, "stats_"):
            self.per_channel_stats()
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        self.stats_.to_csv(out, index=True)
        print(f"[mif-qc] Wrote {out}")

# --- Helper to fix axes ordering if needed ---
def _canonicalize_axes(arr, axes: str, want: str = "CYX"):
    """
    Canonicalize axes by reordering array dimensions.
    Enhanced to handle Z->C conversion for multiplexed data.
    """
    
    # Detect and convert Z->C for multiplexed data
    if _should_convert_z_to_c(axes, arr):
        original_axes = axes
        axes = _convert_z_to_c_axes(axes)
        warnings.warn(f"[MIFQC] Detected multiplexed data: converting {original_axes} -> {axes} "
                     f"({arr.shape[0]} Z slices treated as channels)")
    
    # Validation with error messages
    axes_set = set(axes)
    want_set = set(want)
    
    if axes_set != want_set:
        missing_axes = want_set - axes_set
        extra_axes = axes_set - want_set
        error_msg = f"Unexpected axes string '{axes}'. Expected '{want}'"
        
        if missing_axes:
            error_msg += f". Missing axes: {missing_axes}"
        if extra_axes:
            error_msg += f". Extra axes: {extra_axes}"
            
        raise ValueError(error_msg)

    # Move each letter in `want` to the front in given order
    order: Sequence[int] = []
    for target_axis in want:
        source_idx = axes.index(target_axis)
        order.append(source_idx)
    
    # Transpose the array
    arr_reordered = arr.transpose(order)
    axes_out = "".join(want)
    
    return arr_reordered, axes_out


def _should_convert_z_to_c(axes: str, arr) -> bool:
    """
    Determine if Z axis should be treated as C axis.
    
    This detects multiplexed immunofluorescence data where channels
    are stored as Z slices instead of separate channels.
    
    Args:
        axes: Current axes string (e.g., "ZYX")
        arr: The image array
        
    Returns:
        bool: True if Z should be converted to C
    """
    # Must have Z axis but no C axis
    has_z = 'Z' in axes
    has_c = 'C' in axes
    
    if not has_z or has_c:
        return False
    
    # Get the Z dimension size
    z_axis_idx = axes.index('Z')
    z_size = arr.shape[z_axis_idx]
    
    # Convert if we have multiple Z slices that could be channels
    # Typical range for multiplexed immunofluorescence: 2-50 channels
    return 0 < z_size <= 500


def _convert_z_to_c_axes(axes: str) -> str:
    """
    Convert Z axis to C axis in the axes string.
    
    Args:
        axes: Original axes string (e.g., "ZYX")
        
    Returns:
        str: Modified axes string (e.g., "CYX")
    """
    return axes.replace('Z', 'C')



# --- Helper to extract tiff metadata ---
def _extract_channel_names(tf: TiffFile) -> List[str]:
    try:
        ome = from_tiff(tf.filehandle)
        chs = [c.name or f"ch{i}" for i, c in enumerate(ome.images[0].pixels.channels)]
        if any(ch is None for ch in chs):
            raise ValueError
        return chs
    except Exception:
        # fallbacks …
        ij = tf.imagej_metadata
        if ij and "Labels" in ij:
            return ij["Labels"].split("\n")
        # PageName fallback
        names = [p.tags.get("PageName", None) and p.tags["PageName"].value
                for p in tf.pages]
        if names and all(names):
            return names
    return [f"ch{i}" for i in range(tf.series[0].shape[0])]