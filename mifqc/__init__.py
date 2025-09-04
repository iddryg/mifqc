# mifqc/__init__.py
"""
mIF-QC-Toolkit: Quality control metrics for multiplex immunofluorescence images.

This package provides classes for analyzing:
- Entire images (EntireImage)
- Tiled image regions (TiledImage)  
- Single-cell measurements (CellTable)

Each class computes relevant QC metrics and exports results to CSV files.
"""

__author__ = "Ian Dryg"
__email__ = "iddryg@gmail.com"

# Core classes
from .entire_image import EntireImage
from .tiled_image import TiledImage
from .cell_table import CellTable

# Import other
from typing import List, Optional, Union, Tuple
import warnings
from pathlib import Path

# Core metrics functions
from .qc_metrics import (
    basic_stats,
    gini_index,
    tissue_mask,
)

# Visualization
from .plotting import heatmap

# I/O utilities
from .io_utils import write_report

# CLI app (for programmatic access)
from .cli import app as cli_app

## Define what gets imported with "from mif_qc_toolkit import *"
#__all__ = [
#    # Classes
#    "EntireImage",
#    "TiledImage", 
#    "CellTable",
#    # Metrics
#    "basic_stats",
#    "gini_index",
#    "geary_c",
#    "tissue_mask",
#    # Visualization
#    "heatmap",
#    # I/O
#    "write_report",
#    # CLI
#    "cli_app",
#]

# Convenience imports for common workflows
def quick_whole_image_qc(tiff_path, channel_names=None, output_dir="qc"):
    """
    Quick helper for basic whole-image QC analysis.
    
    Parameters
    ----------
    tiff_path : str or Path
        Path to the OME-TIFF file
    channel_names : list[str], optional
        Names of channels. If None, uses ch0, ch1, etc.
    output_dir : str or Path
        Directory to save CSV results
        
    Returns
    -------
    EntireImage
        The analyzed image object
    """
    from pathlib import Path
    
    img = EntireImage.from_tiff(tiff_path, channel_names=channel_names)
    stats_df = img.per_channel_stats()
    
    # Save results
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img.to_csv(out_dir / f"{img.name}_whole_image_stats.csv")
    
    return img

def quick_tile_analysis(tiff_path, tile_size=512, channel_names=None, output_dir="qc",
                        save_histograms: bool = True, 
                        histogram_bins: int = 100, # default to 100 bins
                        standardized_histogram_range: Optional[Tuple[float, float]] = None) -> TiledImage: # default to uint16 range
    """
    Quick helper for tile-based QC analysis with visualization.

    Parameters
    ----------
    tiff_path : str or Path
        Path to the OME-TIFF file
    tile_size : int
        Size of tiles in pixels
    channel_names : list[str], optional
        Names of channels
    output_dir : str or Path
        Directory to save results
    save_histograms : bool
        If True, calculates, plots, and saves combined histogram data.
    histogram_bins : int
        Number of bins for histograms.
    standardized_histogram_range : tuple[float, float]
        The (min, max) range for all histograms. Defaults to (0, 65535) for uint16 images.

    Returns
    -------
    TiledImage
        The analyzed tiled image object
    """
    from pathlib import Path

    # Load as entire image first
    base_img = EntireImage.from_tiff(tiff_path, channel_names=channel_names)

    # Create tiled version
    tiled = TiledImage(
        base_img.pixels,
        base_img.channel_names,
        name=base_img.name,
        tile_size=tile_size
    )

    # Ensure output directory exists for all results
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Provide a warning at this top-level function if the range is automatically determined
    if standardized_histogram_range is None:
        # The 'tiled' object itself holds the full image pixels and thus has the global_max_intensity property
        warnings.warn(f"Automatically setting histogram range to (0, {tiled.global_max_intensity:.2f}) "
                      f"based on global max pixel value of the entire image.")

    # Compute statistics, now passing the histogram parameters and base output directory
    tile_stats = tiled.tile_statistics(
        save_histograms=save_histograms,
        histogram_bins=histogram_bins,
        standardized_histogram_range=standardized_histogram_range,
        output_base_dir=out_dir, # Pass the output_dir as the base for all generated files
        histogram_output_subdir="histograms" # Default sub-folder for histogram outputs
    )
    summary_stats = tiled.summarize_tiles()

    # Save results (original functionality)
    tiled.tiles_to_csv(out_dir)
    summary_stats.to_csv(out_dir / f"{tiled.name}_tile_summary.csv")

    # Generate heatmaps for first channel (original functionality)
    if tiled.channel_names:
        first_channel = tiled.channel_names
        for metric in ["mean", "std"]:
            if metric in tile_stats.columns:
                heatmap(
                    tile_stats,
                    metric=metric,
                    channel=first_channel,
                    outfile=out_dir / f"{tiled.name}_{first_channel}_{metric}_heatmap.png"
                )

    return tiled

# Convenience functions for single-cell analysis
def quick_cell_table_qc(csv_path, marker_columns=None, output_dir="qc", name=None):
    """
    Quick helper for basic single-cell table QC analysis.
    
    Parameters
    ----------
    csv_path : str or Path
        Path to the CSV file containing single-cell data
    marker_columns : list[str], optional
        Names of marker columns. If None, auto-detects markers.
    output_dir : str or Path
        Directory to save CSV results
    name : str, optional
        Name for the cell table. If None, uses filename.
    
    Returns
    -------
    CellTable
        The analyzed cell table object
    """
    from pathlib import Path
    
    # Load cell table
    table = CellTable.from_csv(csv_path, name=name)
    
    # Calculate statistics
    stats_df = table.per_marker_stats(marker_cols=marker_columns)
    
    # Save results
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.stats_to_csv(out_dir / f"{table.name}_cell_marker_stats.csv")
    
    return table

def quick_tiled_cell_analysis(csv_path, tile_size=1000.0, marker_columns=None, 
                             output_dir="qc", name=None, min_cells_per_tile=10):
    """
    Quick helper for tile-based single-cell QC analysis.
    
    Parameters
    ----------
    csv_path : str or Path
        Path to the CSV file containing single-cell data
    tile_size : float, default=1000.0
        Size of tiles in coordinate units (pixels/microns)
    marker_columns : list[str], optional
        Names of marker columns. If None, auto-detects markers.
    output_dir : str or Path
        Directory to save results
    name : str, optional
        Name for the cell table. If None, uses filename.
    min_cells_per_tile : int, default=10
        Minimum number of cells required per tile for analysis.
    
    Returns
    -------
    TiledCellTable
        The analyzed tiled cell table object
    """
    from pathlib import Path
    
    # Load as regular cell table first
    base_table = CellTable.from_csv(csv_path, name=name)
    
    # Create tiled version
    tiled = TiledCellTable(
        df=base_table.df,
        name=base_table.name,
        marker_columns=marker_columns,
        tile_size=tile_size
    )
    
    # Compute tile statistics
    tile_stats = tiled.tile_statistics(
        marker_cols=marker_columns,
        min_cells_per_tile=min_cells_per_tile
    )
    
    if not tile_stats.empty:
        summary_stats = tiled.summarize_tiles()
        
        # Save results
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        tiled.tiles_to_csv(out_dir)
        tiled.tile_summary_to_csv(out_dir)
        
        print(f"✓ [MIFQC] Processed {len(tile_stats['tile_id'].unique())} tiles")
    else:
        print("⚠ [MIFQC] No tiles met the minimum cell count requirement")
    
    return tiled

# Package info for CLI and debugging
def get_package_info():
    """Return package information for debugging."""
    import sys
    import numpy
    import pandas 
    import dask
    import zarr
    import matplotlib
    
    info = {
        "mif_qc_toolkit": __version__,
        "python": sys.version,
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "dask": dask.__version__,
        "zarr": zarr.__version__,
        "matplotlib": matplotlib.__version__,
    }
    
    try:
        import pysal
        info["pysal"] = pysal.__version__
    except ImportError:
        info["pysal"] = "not installed"
    
    try:
        import libpysal
        info["libpysal"] = libpysal.__version__
    except ImportError:
        info["libpysal"] = "not installed (required for spatial analysis)"
    
    try:
        import esda
        info["esda"] = esda.__version__
    except ImportError:
        info["esda"] = "not installed (required for spatial analysis)"
    
    return info