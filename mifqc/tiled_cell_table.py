# mifqc/tiled_cell_table.py
from __future__ import annotations
import time
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Optional, Union, Dict, Tuple
from dataclasses import dataclass, field
from .cell_table import CellTable

@dataclass
class TiledCellTable(CellTable):
    """
    Spatial tiling of single-cell data based on cell centroid coordinates.
    Similar to TiledImage but for cell-level data instead of pixel data.
    """
    tile_size: float = 1000.0  # Tile size in coordinate units (pixels/microns)
    stride: Optional[float] = None  # Overlap between tiles
    _tile_stats: pd.DataFrame = field(init=False)
    _tile_assignments: pd.Series = field(init=False)
    
    def __post_init__(self):
        """Initialize the tiled cell table."""
        super().__post_init__()
        if self.stride is None:
            self.stride = self.tile_size  # No overlap by default
    
    def _get_tile_bounds(self) -> Tuple[float, float, float, float]:
        """Get the bounds of the cell coordinate space."""
        x_coords = self.df["X_Centroid"].values
        y_coords = self.df["Y_Centroid"].values
        
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        
        return x_min, x_max, y_min, y_max
    
    def _generate_tile_grid(self) -> List[Tuple[float, float, float, float]]:
        """Generate a grid of tiles covering the cell coordinate space."""
        x_min, x_max, y_min, y_max = self._get_tile_bounds()
        
        tiles = []
        y = y_min
        tile_id = 0
        
        while y < y_max:
            x = x_min
            while x < x_max:
                # Define tile boundaries
                x_end = min(x + self.tile_size, x_max)
                y_end = min(y + self.tile_size, y_max)
                
                tiles.append((x, x_end, y, y_end, tile_id))
                tile_id += 1
                x += self.stride
            y += self.stride
        
        return tiles
    
    def _assign_cells_to_tiles(self) -> pd.DataFrame:
        """Assign cells to tiles based on their centroid coordinates."""
        tiles = self._generate_tile_grid()
        
        # Create a mapping of cell indices to tile assignments
        cell_assignments = []
        
        for idx, row in self.df.iterrows():
            x_coord = row["X_Centroid"]
            y_coord = row["Y_Centroid"]
            
            # Find which tiles this cell belongs to
            assigned_tiles = []
            for x_min, x_max, y_min, y_max, tile_id in tiles:
                if x_min <= x_coord < x_max and y_min <= y_coord < y_max:
                    assigned_tiles.append(tile_id)
            
            # Store assignments (a cell can belong to multiple tiles if there's overlap)
            for tile_id in assigned_tiles:
                cell_assignments.append({
                    "cell_index": idx,
                    "tile_id": tile_id,
                    "tile_x_min": [t[0] for t in tiles if t[4] == tile_id][0],
                    "tile_y_min": [t[2] for t in tiles if t[4] == tile_id][0],
                    "tile_x_max": [t[1] for t in tiles if t[4] == tile_id][0],
                    "tile_y_max": [t[3] for t in tiles if t[4] == tile_id][0],
                })
        
        return pd.DataFrame(cell_assignments)
    
    def tile_statistics(self, marker_cols: Optional[List[str]] = None,
                       max_tiles: Optional[int] = None,
                       show_progress: bool = True,
                       min_cells_per_tile: int = 10) -> pd.DataFrame:
        """
        Compute per-tile statistics for single-cell data.
        
        Parameters
        ----------
        marker_cols : List[str], optional
            Marker columns to analyze. If None, uses detected marker columns.
        max_tiles : int, optional
            Maximum number of tiles to process.
        show_progress : bool
            Whether to show progress bar.
        min_cells_per_tile : int, default=10
            Minimum number of cells required per tile for analysis.
            
        Returns
        -------
        pd.DataFrame
            Per-tile statistics with tile information and marker statistics.
        """
        start_time = time.time()
        marker_cols = marker_cols or self.marker_columns
        
        # Get cell-to-tile assignments
        assignments = self._assign_cells_to_tiles()
        
        # Get unique tiles
        unique_tiles = assignments.groupby("tile_id").first().reset_index()
        if max_tiles:
            unique_tiles = unique_tiles.head(max_tiles)
        
        # Progress bar setup
        if show_progress:
            tile_iterator = tqdm(
                unique_tiles.iterrows(),
                total=len(unique_tiles),
                desc=f"Processing {self.name} cell tiles",
                unit="tiles",
                leave=True
            )
        else:
            tile_iterator = unique_tiles.iterrows()
        
        rows = []
        
        for _, tile_info in tile_iterator:
            tile_id = tile_info["tile_id"]
            
            # Get cells in this tile
            cells_in_tile = assignments[assignments["tile_id"] == tile_id]["cell_index"].tolist()
            
            if len(cells_in_tile) < min_cells_per_tile:
                continue  # Skip tiles with too few cells
            
            # Create subset of cells for this tile
            tile_cells = self.df.loc[cells_in_tile]
            
            # Create CellTable for this tile
            tile_cell_table = CellTable(
                df=tile_cells,
                name=f"{self.name}_tile_{tile_id}",
                marker_columns=marker_cols
            )
            
            # Calculate statistics for this tile
            tile_stats = tile_cell_table.per_marker_stats(show_progress=False)
            
            # Add tile information to each marker's stats
            for marker, stats in tile_stats.iterrows():
                stats_dict = stats.to_dict()
                stats_dict.update({
                    "tile_id": tile_id,
                    "tile_x_min": tile_info["tile_x_min"],
                    "tile_y_min": tile_info["tile_y_min"],
                    "tile_x_max": tile_info["tile_x_max"],
                    "tile_y_max": tile_info["tile_y_max"],
                    "tile_center_x": (tile_info["tile_x_min"] + tile_info["tile_x_max"]) / 2,
                    "tile_center_y": (tile_info["tile_y_min"] + tile_info["tile_y_max"]) / 2,
                    "n_cells": len(cells_in_tile),
                    "cell_density": len(cells_in_tile) / (self.tile_size ** 2),
                    "marker": marker
                })
                rows.append(stats_dict)
            
            # Update progress
            if show_progress and hasattr(tile_iterator, 'set_postfix'):
                tile_iterator.set_postfix({
                    'tile_id': tile_id,
                    'n_cells': len(cells_in_tile),
                    'markers': len(marker_cols)
                })
        
        if not rows:
            warnings.warn("No tiles met the minimum cell count requirement")
            return pd.DataFrame()
        
        self._tile_stats = pd.DataFrame(rows)
        
        # Calculate timing
        elapsed_time = time.time() - start_time
        if show_progress:
            print(f"✓ [MIFQC] Processed {len(unique_tiles)} tiles in {elapsed_time:.2f}s "
                  f"({elapsed_time/len(unique_tiles):.3f}s per tile)")
        
        return self._tile_stats
    
    def summarize_tiles(self) -> pd.DataFrame:
        """Aggregate statistics across all tiles."""
        if not hasattr(self, "_tile_stats") or self._tile_stats.empty:
            self.tile_statistics()
        
        if self._tile_stats.empty:
            return pd.DataFrame()
        
        # Group by marker and calculate summary statistics
        grouped = self._tile_stats.groupby("marker").agg({
            "mean": ["mean", "std", "min", "max"],
            "std": ["mean", "std", "min", "max"],
            "gini": ["mean", "std", "min", "max"],
            "median": ["mean", "std", "min", "max"],
            "cv": ["mean", "std", "min", "max"],
            "n_cells": ["mean", "std", "min", "max"],
            "cell_density": ["mean", "std", "min", "max"]
        })
        
        # Flatten multi-index columns
        grouped.columns = ["_".join(c) for c in grouped.columns]
        
        return grouped
    
    def get_tile_assignments(self) -> pd.DataFrame:
        """Get the cell-to-tile assignment table."""
        if not hasattr(self, "_tile_assignments"):
            self._tile_assignments = self._assign_cells_to_tiles()
        return self._tile_assignments
    
    def get_cells_in_tile(self, tile_id: int) -> pd.DataFrame:
        """Get all cells assigned to a specific tile."""
        assignments = self.get_tile_assignments()
        cell_indices = assignments[assignments["tile_id"] == tile_id]["cell_index"].tolist()
        return self.df.loc[cell_indices]
    
    # cells per area per tile per phenotype
    def calculate_cell_density_overall_per_tile(self, method='convex_hull', 
                                            include_phenotypes=True,
                                            buffer_fraction=0.05) -> pd.DataFrame:
        """
        Calculate overall cell density for each tile in the tiled cell table.
        
        Parameters
        ----------
        method : str, default='convex_hull'
            Method to create tissue mask for each tile.
        include_phenotypes : bool, default=True
            Whether to calculate densities for individual phenotypes.
        buffer_fraction : float, default=0.05
            Fraction of coordinate range to use as buffer.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with density statistics for each tile.
        """
        if not hasattr(self, "_tile_assignments"):
            assignments = self._assign_cells_to_tiles()
        else:
            assignments = self._tile_assignments
        
        # Get unique tiles
        unique_tiles = assignments.groupby("tile_id").first().reset_index()
        
        results = []
        
        for _, tile_info in unique_tiles.iterrows():
            tile_id = tile_info["tile_id"]
            
            # Get cells in this tile
            cells_in_tile = assignments[assignments["tile_id"] == tile_id]["cell_index"].tolist()
            
            if len(cells_in_tile) == 0:
                continue
            
            # Create subset of cells for this tile
            tile_cells = self.df.loc[cells_in_tile]
            
            # Create temporary CellTable for this tile
            from .cell_table import CellTable
            tile_cell_table = CellTable(
                df=tile_cells,
                name=f"{self.name}_tile_{tile_id}"
            )
            
            # Calculate density for this tile
            tile_density = tile_cell_table.calculate_cell_density_overall(
                method=method,
                include_phenotypes=include_phenotypes,
                buffer_fraction=buffer_fraction
            )
            
            # Add tile information
            tile_density.update({
                "tile_id": tile_id,
                "tile_x_min": tile_info["tile_x_min"],
                "tile_y_min": tile_info["tile_y_min"],
                "tile_x_max": tile_info["tile_x_max"],
                "tile_y_max": tile_info["tile_y_max"],
                "tile_center_x": (tile_info["tile_x_min"] + tile_info["tile_x_max"]) / 2,
                "tile_center_y": (tile_info["tile_y_min"] + tile_info["tile_y_max"]) / 2,
            })
            
            results.append(tile_density)
        
        return pd.DataFrame(results) if results else pd.DataFrame()

    # ----------- Export functions -----------
    def tiles_to_csv(self, folder: Union[str, Path]):
        """Export per-tile statistics to CSV."""
        if not hasattr(self, "_tile_stats") or self._tile_stats.empty:
            self.tile_statistics()
        
        if self._tile_stats.empty:
            warnings.warn("No tile statistics to export")
            return
        
        out = Path(folder) / f"{self.name}_per_tile_cells.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        self._tile_stats.to_csv(out, index=False)
        print(f"[mif-qc] Wrote {out}")
    
    def tile_summary_to_csv(self, folder: Union[str, Path]):
        """Export tile summary statistics to CSV."""
        summary = self.summarize_tiles()
        if summary.empty:
            warnings.warn("No tile summary to export")
            return
        
        out = Path(folder) / f"{self.name}_tile_summary_cells.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out, index=True)
        print(f"[mif-qc] Wrote {out}")