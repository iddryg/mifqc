# mifqc/qc_metrics.py
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import filters, morphology
from esda.geary import Geary
import libpysal  # for spatial weights
import warnings

__all__ = [
    "basic_stats",
    "gini_index",
    "geary_c",
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

# ---------- Geary's C (replaces Moran's I) ----------
def geary_c(arr: np.ndarray, downsample_factor: int = 4, max_pixels: int = 1_000_000) -> float:
    """
    Global Geary's C for a single 2-D channel with memory management.
    
    Geary's C is less computationally intensive than Moran's I and better
    suited for large images in quality control workflows.
    
    Parameters
    ----------
    arr : np.ndarray
        Input 2D image array
    downsample_factor : int, default=4
        Factor by which to downsample the image if it's too large
    max_pixels : int, default=1_000_000
        Maximum number of pixels to process without downsampling
        
    Returns
    -------
    float
        Geary's C value. Values close to 1 indicate spatial randomness,
        values < 1 indicate positive spatial autocorrelation (clustering),
        values > 1 indicate negative spatial autocorrelation (dispersion).
    """
    try:
        # Memory management: downsample large images
        if arr.size > max_pixels:
            # Downsample to reduce memory usage
            from skimage.transform import downscale_local_mean
            factor = max(1, int(np.sqrt(arr.size / max_pixels)))
            arr_processed = downscale_local_mean(arr, (factor, factor)).astype(float)
            warnings.warn(
                f"Image downsampled by factor {factor} for Geary's C calculation "
                f"(original: {arr.shape}, processed: {arr_processed.shape})"
            )
        else:
            arr_processed = arr.astype(float)
        
        y = arr_processed.ravel()
        rows, cols = arr_processed.shape
        
        # Build queen-contiguity weights on the image grid
        # This is the same spatial weights as Moran's I but Geary's C is more efficient
        w = libpysal.weights.lat2W(rows, cols, rook=False)
        
        # Calculate Geary's C - this is computationally lighter than Moran's I
        gc = Geary(y, w, permutations=0)  # Skip permutations to save time
        
        return gc.C
        
    except MemoryError:
        warnings.warn(
            f"MemoryError computing Geary's C for image shape {arr.shape}. "
            "Returning NaN. Consider using tile-based analysis for large images."
        )
        return np.nan
    except Exception as e:
        warnings.warn(f"Error computing Geary's C: {e}. Returning NaN.")
        return np.nan

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
