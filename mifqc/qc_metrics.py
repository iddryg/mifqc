# mifqc/qc_metrics.py
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import filters, morphology
from esda.moran import Moran
import libpysal  # for spatial weights

__all__ = [
    "basic_stats",
    "gini_index",
    "moran_i",
    "tissue_mask",
]

# ---------- Basic per-channel statistics ----------
def basic_stats(arr: np.ndarray) -> dict:
    """Return min, max, mean, std for a 2-D (single-channel) image."""
    return dict(
        min=float(arr.min()),
        max=float(arr.max()),
        mean=float(arr.mean()),
        std=float(arr.std(ddof=1)),
    )

# ---------- Gini index ----------
def gini_index(arr: np.ndarray) -> float:
    """Gini index of pixel intensities."""
    flat = arr.flatten().astype(float)
    if np.issubdtype(flat.dtype, np.integer):
        flat = flat + 1e-9  # avoid zeros
    flat.sort()
    n = flat.size
    cum = np.cumsum(flat, dtype=float)
    return (n + 1 - 2 * cum.sum() / cum[-1]) / n

# ---------- Moran’s I ----------
def moran_i(arr: np.ndarray, normalize: bool = True) -> float:
    """Global Moran’s I for a single 2-D channel."""
    y = arr.astype(float).ravel()
    # Build queen-contiguity weights on the image grid
    rows, cols = arr.shape
    w = libpysal.weights.lat2W(rows, cols, rook=False)
    mi = Moran(y, w, two_tailed=False)
    return mi.I if normalize else mi.I * w.s0  # raw numerator if needed

# ---------- Tissue mask ----------
def tissue_mask(
    dapi: np.ndarray,
    otsu_ratio: float = 1.0,
    min_size: int = 256,
) -> np.ndarray:
    """
    Binary mask of 'tissue' pixels from a nuclear (DAPI) channel.

    Parameters
    ----------
    dapi : 2-D array
        DAPI channel image.
    otsu_ratio : float
        Factor multiplying Otsu threshold.
    min_size : int
        Remove connected components smaller than this (#pixels).
    """
    thr = filters.threshold_otsu(dapi) * otsu_ratio
    mask = dapi > thr
    mask = morphology.remove_small_objects(mask, min_size=min_size)
    mask = morphology.remove_small_holes(mask, area_threshold=min_size)
    return mask.astype(bool)
