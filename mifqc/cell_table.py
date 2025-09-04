# mifqc/cell_table.py
from __future__ import annotations
import pandas as pd
from pathlib import Path
from typing import List, Optional, Union
from dataclasses import dataclass, field
from .qc_metrics import gini_index

@dataclass
class CellTable:
    df: pd.DataFrame                            # must include X_Centroid, Y_Centroid
    name: str = "cells"

    # ---------- Column-wise univariate descriptors ----------
    def per_marker_stats(self, marker_cols: Optional[List[str]] = None) -> pd.DataFrame:
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

    # ---------- Export ----------
    def to_csv(self, out_path: Union[str, Path]):
        self.df.to_csv(out_path, index=False)
