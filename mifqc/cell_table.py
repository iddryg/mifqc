# mifqc/cell_table.py
from __future__ import annotations
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from .qc_metrics import gini_index, moran_i

@dataclass
class CellTable:
    df: pd.DataFrame                            # must include X_Centroid, Y_Centroid
    name: str = "cells"

    # ---------- Column-wise univariate descriptors ----------
    def per_marker_stats(self, marker_cols: list[str] | None = None) -> pd.DataFrame:
        marker_cols = marker_cols or [
            c for c in self.df.columns if c not in ("CellID", "X_Centroid", "Y_Centroid")
        ]
        rows = []
        for col in marker_cols:
            vals = self.df[col].values
            rows.append(
                dict(
                    marker=col,
                    mean=float(vals.mean()),
                    std=float(vals.std(ddof=1)),
                    gini=gini_index(vals),
                )
            )
        return pd.DataFrame(rows).set_index("marker")

    # ---------- Spatial autocorrelation at cellular scale ----------
    def moran_per_marker(self, marker_cols: list[str] | None = None) -> pd.Series:
        import libpysal
        marker_cols = marker_cols or [
            c for c in self.df.columns if c not in ("CellID", "X_Centroid", "Y_Centroid")
        ]
        # Build K-nearest neighbor weights (k=8)
        coords = self.df[["X_Centroid", "Y_Centroid"]].values
        w = libpysal.weights.KNN.from_array(coords, k=8)
        w.transform = "r"
        out = {}
        for col in marker_cols:
            mi = moran_i(self.df[col].values.reshape(-1, 1), normalize=True)
            out[col] = mi
        return pd.Series(out, name="Moran_I")

    # ---------- Export ----------
    def to_csv(self, out_path: str | Path):
        self.df.to_csv(out_path, index=False)
