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

    # ---------- Spatial autocorrelation at cellular scale ----------
    def geary_c_per_marker(self, marker_cols: Optional[List[str]] = None, k: int = 8) -> pd.Series:
        """
        Calculate Geary's C for each marker using cell-level data.
        
        Geary's C is more computationally efficient than Moran's I and better
        suited for large cell datasets in quality control workflows.
        
        Parameters
        ----------
        marker_cols : List[str], optional
            Marker columns to analyze. If None, uses all non-coordinate columns.
        k : int, default=8
            Number of nearest neighbors to consider for spatial weights.
            
        Returns
        -------
        pd.Series
            Geary's C values for each marker. Values close to 1 indicate spatial randomness,
            values < 1 indicate positive spatial autocorrelation (clustering),
            values > 1 indicate negative spatial autocorrelation (dispersion).
        """
        import libpysal
        from esda.geary import Geary
        marker_cols = marker_cols or [
            c for c in self.df.columns if c not in ("CellID", "X_Centroid", "Y_Centroid")
        ]
        
        if len(self.df) < k + 1:
            warnings.warn(f"Not enough cells ({len(self.df)}) for k={k} neighbors. Returning NaN values.")
            return pd.Series({col: np.nan for col in marker_cols}, name="Geary_C")
        
        try:
            # Build K-nearest neighbor weights
            coords = self.df[["X_Centroid", "Y_Centroid"]].values
            w = libpysal.weights.KNN.from_array(coords, k=k)
            w.transform = "r"  # Row-standardized weights
            
            out = {}
            for col in marker_cols:
                try:
                    # Get marker values
                    y = self.df[col].values.astype(float)
                    
                    # Check for valid data
                    if len(y) == 0 or np.all(np.isnan(y)) or np.var(y) == 0:
                        out[col] = np.nan
                        continue
                    
                    # Calculate Geary's C using esda.Geary (designed for point data)
                    gc = Geary(y, w, permutations=0)  # Skip permutations for speed
                    out[col] = gc.C
                    
                except Exception as e:
                    warnings.warn(f"Error computing Geary's C for marker {col}: {e}")
                    out[col] = np.nan
            
            return pd.Series(out, name="Geary_C")
            
        except Exception as e:
            warnings.warn(f"Error in spatial weights construction: {e}")
            return pd.Series({col: np.nan for col in marker_cols}, name="Geary_C")

    # ---------- Export ----------
    def to_csv(self, out_path: Union[str, Path]):
        self.df.to_csv(out_path, index=False)
