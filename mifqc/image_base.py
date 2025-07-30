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
    #def from_tiff(cls, path: Union[str, Path], channel_names: Optional[List[str]] = None):
    #    arr = imread(path)
    #    if arr.ndim == 2:          # single channel TIFF
    #        arr = arr[None, ...]
    #    if channel_names is None:
    #        channel_names = [f"ch{i}" for i in range(arr.shape[0])]
    #    return cls(arr, channel_names, name=Path(path).stem)
    #
    def from_tiff(cls, path: Union[str, Path], axes: Optional[str] = None,
              channel_names: Optional[List[str]] = None):
        with TiffFile(path) as tf:
            ser = tf.series[0]          # handle multi-series separately if needed
            axes_in = ser.axes if axes is None else axes
            arr = ser.asarray()         # honours memory-mapping or zarr if selected
        if axes_in != "CYX":
            arr, axes_out = _canonicalize_axes(arr, axes_in, want="CYX")
            warnings.warn(f"Reordered axes {axes_in} → {axes_out}")
        if channel_names is None:
            channel_names = _extract_channel_names(tf)  # see below
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


    # --- Helper to fix axes ordering if needed ---
    def _canonicalize_axes(
        arr: ArrayLike, axes: str, want: str = "CYX"
        ) -> Tuple[ArrayLike, str]:
        """
        Reorder `arr` so that its axes become `want` (default "CYX…").

        Parameters
        ----------
        arr   : numpy.ndarray | dask.array
        axes  : str           Current order returned by tifffile.series[0].axes
        want  : str           Desired leading axes. Remainder keeps incoming order.

        Returns
        -------
        arr_reordered, new_axes
        """
        # sanity
        if set(axes) != set(want) | set(axes) - set(want):
            raise ValueError(f"Unexpected axes string {axes}")

        # Move each letter in `want` to the front in given order
        order: Sequence[int] = []
        for ax in want:
            try:
                order.append(axes.index(ax))
            except ValueError:
                raise ValueError(f"Required axis {ax} not found in {axes}")

        # keep remaining axes in original order
        order.extend(i for i, ax in enumerate(axes) if ax not in want)

        moved = da.moveaxis(arr, range(len(order)), order) if hasattr(arr, "chunks") \
                else np.moveaxis(arr, range(len(order)), order)
        new_axes = "".join(axes[i] for i in order)
        return moved, new_axes
    

    # --- Helper to extract tiff metadata ---
    def _extract_channel_names(tf: TiffFile) -> list[str]:
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


    # ---------- Pixel-wise statistics ----------
    def per_channel_stats(self) -> pd.DataFrame:
        rows: list[dict] = []
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
