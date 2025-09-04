# mifqc/cell_table.py
from __future__ import annotations
import time
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Optional, Union, Dict, Tuple
from dataclasses import dataclass, field
from .qc_metrics import gini_index, basic_stats

@dataclass
class CellTable:
    """Single-cell data table with QC metrics and spatial analysis capabilities."""
    df: pd.DataFrame  # must include X_Centroid, Y_Centroid
    name: str = "cells"
    marker_columns: Optional[List[str]] = field(default=None)
    stats_: pd.DataFrame = field(init=False)
    
    def __post_init__(self):
        """Validate and set up the cell table after initialization."""
        self._validate_required_columns()
        if self.marker_columns is None:
            self.marker_columns = self._detect_marker_columns()
    
    def _validate_required_columns(self):
        """Check for required columns in the dataframe."""
        required_columns = ["X_Centroid", "Y_Centroid"]
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    def _detect_marker_columns(self) -> List[str]:
        """Automatically detect marker columns (exclude coordinate and ID columns)."""
        exclude_patterns = [
            "CellID", "Cell_ID", "ID", "ObjectNumber",
            "X_Centroid", "Y_Centroid", "X_Position", "Y_Position",
            "Centroid_X", "Centroid_Y", "Location_Center_X", "Location_Center_Y",
            "UMAP", "PCA", "tSNE", "PC1", "PC2", "PC3"
        ]
        
        marker_cols = []
        for col in self.df.columns:
            # Skip columns that match exclusion patterns
            if any(pattern in col for pattern in exclude_patterns):
                continue
            # Skip obvious morphology columns
            if any(morph in col.lower() for morph in ["area", "perimeter", "eccentricity", "solidity", "extent"]):
                continue
            marker_cols.append(col)
        
        return marker_cols
    
    # ----------- Column-wise univariate descriptors -----------
    def per_marker_stats(self, marker_cols: Optional[List[str]] = None, 
                        show_progress: bool = True) -> pd.DataFrame:
        """
        Calculate comprehensive statistics for each marker across all cells.
        
        Parameters
        ----------
        marker_cols : List[str], optional
            Marker columns to analyze. If None, uses detected marker columns.
        show_progress : bool
            Whether to show progress bar.
            
        Returns
        -------
        pd.DataFrame
            Statistics for each marker with marker names as index.
        """
        start_time = time.time()
        marker_cols = marker_cols or self.marker_columns
        
        if show_progress and len(marker_cols) > 1:
            marker_iterator = tqdm(
                marker_cols,
                desc=f"Analyzing {self.name} markers",
                unit="markers",
                leave=True
            )
        else:
            marker_iterator = marker_cols
        
        rows = []
        for col in marker_iterator:
            if col not in self.df.columns:
                warnings.warn(f"Column {col} not found in dataframe")
                continue
                
            vals = self.df[col].values
            
            # Calculate comprehensive statistics
            stats_dict = basic_stats(vals)
            stats_dict.update({
                "gini": gini_index(vals),
                "median": float(np.median(vals)),
                "q25": float(np.percentile(vals, 25)),
                "q75": float(np.percentile(vals, 75)),
                "iqr": float(np.percentile(vals, 75) - np.percentile(vals, 25)),
                "cv": float(np.std(vals) / np.mean(vals)) if np.mean(vals) != 0 else np.nan,
                "skewness": float(self._calculate_skewness(vals)),
                "kurtosis": float(self._calculate_kurtosis(vals)),
                "zero_fraction": float(np.sum(vals == 0) / len(vals)),
                "marker": col
            })
            rows.append(stats_dict)
        
        self.stats_ = pd.DataFrame(rows).set_index("marker")
        
        elapsed_time = time.time() - start_time
        if show_progress:
            print(f"✓ [MIFQC] Analyzed {len(marker_cols)} markers in {elapsed_time:.2f}s "
                  f"({elapsed_time/len(marker_cols):.3f}s per marker)")
        
        return self.stats_
    
    @staticmethod
    def _calculate_skewness(vals: np.ndarray) -> float:
        """Calculate skewness using the third moment."""
        vals = vals[~np.isnan(vals)]  # Remove NaN values
        if len(vals) < 3:
            return np.nan
        mean_val = np.mean(vals)
        std_val = np.std(vals, ddof=1)
        if std_val == 0:
            return 0.0
        return np.mean(((vals - mean_val) / std_val) ** 3)
    
    @staticmethod
    def _calculate_kurtosis(vals: np.ndarray) -> float:
        """Calculate excess kurtosis using the fourth moment."""
        vals = vals[~np.isnan(vals)]  # Remove NaN values
        if len(vals) < 4:
            return np.nan
        mean_val = np.mean(vals)
        std_val = np.std(vals, ddof=1)
        if std_val == 0:
            return 0.0
        return np.mean(((vals - mean_val) / std_val) ** 4) - 3
    
    # Simple cell density measures
    def _detect_phenotype_columns(self) -> List[str]:
        """
        Detect which columns represent cell phenotypes (boolean markers).
        
        Returns
        -------
        List[str]
            List of column names that appear to be phenotype labels.
        """
        phenotype_cols = []
        
        for col in self.df.columns:
            # Skip coordinate and ID columns
            if col in ["X_Centroid", "Y_Centroid", "CellID", "Cell_ID", "ID", "ObjectNumber"]:
                continue
            
            # Check if column contains only boolean-like values
            unique_vals = self.df[col].dropna().unique()
            
            # Boolean phenotype indicators
            is_boolean = (
                # True boolean values
                set(unique_vals).issubset({True, False}) or
                # 0/1 binary values
                set(unique_vals).issubset({0, 1, 0.0, 1.0}) or
                # String boolean values
                set(str(v).lower() for v in unique_vals).issubset({'true', 'false', '0', '1'})
            )
            
            # Also check column naming patterns common for phenotypes
            phenotype_patterns = ['+', '-', '_pos', '_neg', 'phenotype', 'cluster', 'type']
            has_phenotype_pattern = any(pattern in col.lower() for pattern in phenotype_patterns)
            
            if is_boolean or has_phenotype_pattern:
                # Double-check it's not a morphology column
                morph_patterns = ['area', 'perimeter', 'eccentricity', 'solidity', 'extent']
                if not any(morph in col.lower() for morph in morph_patterns):
                    phenotype_cols.append(col)
        
        return phenotype_cols

    def _detect_marker_intensity_columns(self) -> List[str]:
        """
        Detect which columns represent marker fluorescence intensities (continuous numeric).
        
        Returns
        -------
        List[str]
            List of column names that appear to be fluorescence intensity measurements.
        """
        intensity_cols = []
        
        for col in self.df.columns:
            # Skip coordinate, ID, and phenotype columns
            if col in ["X_Centroid", "Y_Centroid", "CellID", "Cell_ID", "ID", "ObjectNumber"]:
                continue
            
            phenotype_cols = self._detect_phenotype_columns()
            if col in phenotype_cols:
                continue
            
            # Skip obvious morphology columns
            morph_patterns = ['area', 'perimeter', 'eccentricity', 'solidity', 'extent', 'compactness']
            if any(morph in col.lower() for morph in morph_patterns):
                continue
            
            # Skip dimensionality reduction columns
            dimred_patterns = ['umap', 'pca', 'tsne', 'pc1', 'pc2', 'pc3']
            if any(dimred in col.lower() for dimred in dimred_patterns):
                continue
            
            # Check if column contains continuous numeric values
            try:
                values = pd.to_numeric(self.df[col], errors='coerce')
                non_null_values = values.dropna()
                
                if len(non_null_values) > 0:
                    # Check if values are continuous (not just 0/1)
                    unique_vals = non_null_values.unique()
                    if len(unique_vals) > 2:  # More than just binary values
                        intensity_cols.append(col)
                    elif len(unique_vals) == 2 and not set(unique_vals).issubset({0, 1, 0.0, 1.0}):
                        # Two values but not 0/1, likely intensity
                        intensity_cols.append(col)
            except:
                continue
        
        return intensity_cols

    def _create_tissue_mask(self, method='convex_hull', buffer_fraction=0.05):
        """
        Create a mask representing the tissue area occupied by cells.
        
        Parameters
        ----------
        method : str, default='convex_hull'
            Method to create the mask. Options: 'convex_hull', 'bounding_box'
        buffer_fraction : float, default=0.05
            Fraction of coordinate range to use as buffer around cells.
        
        Returns
        -------
        tuple
            (mask_area, mask_bounds) where mask_area is the area and mask_bounds are the boundaries.
        """
        coords = self.df[["X_Centroid", "Y_Centroid"]].values
        
        if len(coords) < 3:
            # Not enough points for a proper mask, use bounding box
            method = 'bounding_box'
        
        x_coords, y_coords = coords[:, 0], coords[:, 1]
        x_range = x_coords.max() - x_coords.min()
        y_range = y_coords.max() - y_coords.min()
        
        if method == 'convex_hull':
            try:
                from scipy.spatial import ConvexHull
                
                # Create convex hull
                hull = ConvexHull(coords)
                mask_area = hull.volume  # In 2D, volume gives the area
                
                # Get hull boundaries
                hull_points = coords[hull.vertices]
                mask_bounds = {
                    'x_min': hull_points[:, 0].min(),
                    'x_max': hull_points[:, 0].max(),
                    'y_min': hull_points[:, 1].min(),
                    'y_max': hull_points[:, 1].max(),
                    'method': 'convex_hull',
                    'hull_points': hull_points
                }
                
            except ImportError:
                warnings.warn("scipy not available, falling back to bounding box method")
                method = 'bounding_box'
            except Exception as e:
                warnings.warn(f"Convex hull calculation failed: {e}, falling back to bounding box")
                method = 'bounding_box'
        
        if method == 'bounding_box':
            # Simple bounding box with buffer
            buffer_x = x_range * buffer_fraction
            buffer_y = y_range * buffer_fraction
            
            x_min = x_coords.min() - buffer_x
            x_max = x_coords.max() + buffer_x
            y_min = y_coords.min() - buffer_y
            y_max = y_coords.max() + buffer_y
            
            mask_area = (x_max - x_min) * (y_max - y_min)
            mask_bounds = {
                'x_min': x_min,
                'x_max': x_max,
                'y_min': y_min,
                'y_max': y_max,
                'method': 'bounding_box'
            }
        
        return mask_area, mask_bounds

    def calculate_cell_density_overall(self, method='convex_hull', 
                                    include_phenotypes=True, 
                                    buffer_fraction=0.05) -> Dict[str, float]:
        """
        Calculate overall cell density within the tissue area occupied by cells.
        
        This function creates a mask around all cells to define the tissue area,
        then calculates cell density as cells per unit area. Also calculates
        density for each phenotype if boolean phenotype columns are present.
        
        Parameters
        ----------
        method : str, default='convex_hull'
            Method to create tissue mask. Options: 'convex_hull', 'bounding_box'
        include_phenotypes : bool, default=True
            Whether to calculate densities for individual phenotypes.
        buffer_fraction : float, default=0.05
            Fraction of coordinate range to use as buffer (for bounding_box method).
        
        Returns
        -------
        Dict[str, float]
            Dictionary with density values. Keys include 'total_density' and 
            individual phenotype densities if available.
        """
        if len(self.df) == 0:
            return {"total_density": 0.0, "tissue_area": 0.0, "total_cells": 0}
        
        # Create tissue mask
        mask_area, mask_bounds = self._create_tissue_mask(method=method, buffer_fraction=buffer_fraction)
        
        # Calculate total cell density
        total_cells = len(self.df)
        total_density = total_cells / mask_area if mask_area > 0 else 0.0
        
        results = {
            "total_density": total_density,
            "tissue_area": mask_area,
            "total_cells": total_cells,
            "mask_method": mask_bounds['method']
        }
        
        # Add mask boundary information
        for key in ['x_min', 'x_max', 'y_min', 'y_max']:
            if key in mask_bounds:
                results[f"mask_{key}"] = mask_bounds[key]
        
        # Calculate phenotype-specific densities
        if include_phenotypes:
            phenotype_cols = self._detect_phenotype_columns()
            
            for pheno_col in phenotype_cols:
                try:
                    # Count positive cells for this phenotype
                    positive_cells = self.df[pheno_col].sum() if pheno_col in self.df.columns else 0
                    pheno_density = positive_cells / mask_area if mask_area > 0 else 0.0
                    
                    results[f"{pheno_col}_density"] = pheno_density
                    results[f"{pheno_col}_count"] = int(positive_cells)
                    results[f"{pheno_col}_fraction"] = positive_cells / total_cells if total_cells > 0 else 0.0
                    
                except Exception as e:
                    warnings.warn(f"Could not calculate density for phenotype {pheno_col}: {e}")
                    results[f"{pheno_col}_density"] = np.nan
        
        return results

    def get_column_summary(self) -> Dict[str, List[str]]:
        """
        Get a summary of different column types in the cell table.
        
        Returns
        -------
        Dict[str, List[str]]
            Dictionary with column types as keys and lists of column names as values.
        """
        summary = {
            'coordinate_columns': [col for col in ['X_Centroid', 'Y_Centroid'] if col in self.df.columns],
            'marker_intensity_columns': self._detect_marker_intensity_columns(),
            'phenotype_columns': self._detect_phenotype_columns(),
            'other_columns': []
        }
        
        # Find other columns
        all_detected = (summary['coordinate_columns'] + 
                    summary['marker_intensity_columns'] + 
                    summary['phenotype_columns'])
        
        summary['other_columns'] = [col for col in self.df.columns if col not in all_detected]
        
        return summary

    def plot_tissue_mask(self, method='convex_hull', buffer_fraction=0.05, 
                        figsize=(10, 8), save_path=None):
        """
        Plot the cells and the calculated tissue mask for visualization.
        
        Parameters
        ----------
        method : str, default='convex_hull'
            Method to create tissue mask.
        buffer_fraction : float, default=0.05
            Buffer fraction for bounding box method.
        figsize : tuple, default=(10, 8)
            Figure size for the plot.
        save_path : str or Path, optional
            Path to save the plot. If None, displays the plot.
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            
            # Plot all cells
            coords = self.df[["X_Centroid", "Y_Centroid"]].values
            ax.scatter(coords[:, 0], coords[:, 1], alpha=0.6, s=1, c='blue', label='Cells')
            
            # Get mask information
            mask_area, mask_bounds = self._create_tissue_mask(method=method, buffer_fraction=buffer_fraction)
            
            # Plot the mask
            if mask_bounds['method'] == 'convex_hull' and 'hull_points' in mask_bounds:
                # Plot convex hull
                hull_points = mask_bounds['hull_points']
                # Close the hull by adding the first point at the end
                hull_closed = np.vstack([hull_points, hull_points[0]])
                ax.plot(hull_closed[:, 0], hull_closed[:, 1], 'r-', linewidth=2, label='Convex Hull')
                ax.fill(hull_closed[:, 0], hull_closed[:, 1], alpha=0.2, color='red')
            
            elif mask_bounds['method'] == 'bounding_box':
                # Plot bounding box
                x_min, x_max = mask_bounds['x_min'], mask_bounds['x_max']
                y_min, y_max = mask_bounds['y_min'], mask_bounds['y_max']
                
                rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                    linewidth=2, edgecolor='red', facecolor='red', alpha=0.2,
                                    label='Bounding Box')
                ax.add_patch(rect)
            
            ax.set_xlabel('X Coordinate')
            ax.set_ylabel('Y Coordinate')
            ax.set_title(f'Tissue Mask Visualization ({mask_bounds["method"]})\nArea: {mask_area:.2f}, Cells: {len(coords)}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"[mif-qc] Saved tissue mask plot to {save_path}")
            else:
                plt.show()
            
            plt.close()
            
        except ImportError:
            warnings.warn("matplotlib not available for plotting")
        except Exception as e:
            warnings.warn(f"Could not create tissue mask plot: {e}")

    # ----------- Cell density and neighborhood analysis -----------
    def calculate_cell_density(self, radius: float = 100.0) -> pd.Series:
        """
        Calculate local cell density for each cell within a given radius.
        
        Parameters
        ----------
        radius : float, default=100.0
            Radius in pixels/microns for density calculation.
            
        Returns
        -------
        pd.Series
            Cell density values for each cell.
        """
        from scipy.spatial.distance import cdist
        
        coords = self.df[["X_Centroid", "Y_Centroid"]].values
        
        # Calculate pairwise distances
        distances = cdist(coords, coords)
        
        # Count neighbors within radius for each cell
        densities = []
        for i in range(len(coords)):
            # Count cells within radius (excluding self)
            neighbors = np.sum((distances[i] <= radius) & (distances[i] > 0))
            # Normalize by area (π * r²)
            area = np.pi * radius ** 2
            density = neighbors / area
            densities.append(density)
        
        return pd.Series(densities, name="cell_density")
    
    # ----------- Export functions -----------    
    def to_csv(self, out_path: Union[str, Path]):
        """Export cell table to CSV."""
        self.df.to_csv(out_path, index=False)
        print(f"[mif-qc] Wrote {out_path}")
    
    def stats_to_csv(self, out_path: Union[str, Path]):
        """Export marker statistics to CSV."""
        if not hasattr(self, "stats_"):
            self.per_marker_stats()
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        self.stats_.to_csv(out, index=True)
        print(f"[mif-qc] Wrote {out}")

    @classmethod
    def from_csv(cls, file_path: Union[str, Path], name: Optional[str] = None):
        """Load cell table from CSV file."""
        df = pd.read_csv(file_path)
        table_name = name or Path(file_path).stem
        return cls(df=df, name=table_name)