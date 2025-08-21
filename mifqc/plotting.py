# mifqc/plotting.py
from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, Optional, List, Dict, Tuple
import warnings

def heatmap(df: pd.DataFrame, metric: str, channel: str,
            outfile: Optional[Union[str, Path]] = None, cmap="viridis"):
    """
    Draw a per-tile heat-map for a given metric and channel.

    Parameters
    ----------
    df        : DataFrame returned by `TiledImage.tile_statistics()`
    metric    : column to visualise (e.g. "mean", "geary_c")
    channel   : channel name present in df["channel"]
    """
    sub = df[df["channel"] == channel]
    pivot = sub.pivot(index="tile_y", columns="tile_x", values=metric)
    plt.figure(figsize=(6, 5))
    plt.pcolormesh(pivot.values[::-1], cmap=cmap, shading="auto")
    plt.title(f"{metric} – {channel}")
    plt.colorbar(label=metric)
    plt.axis("off")
    if outfile:
        Path(outfile).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outfile, dpi=300, bbox_inches="tight")
    return plt.gca()


def plot_histogram(counts: np.ndarray, bin_edges: np.ndarray,
                   outfile: Optional[Union[str, Path]] = None,
                   title: str = "Pixel Intensity Histogram"):
    """
    Draws a histogram of pixel intensities.

    Parameters
    ----------
    counts : np.ndarray
        The counts of pixels in each bin.
    bin_edges : np.ndarray
        The edges of the histogram bins.
    outfile : str or Path, optional
        Path to save the histogram plot. If None, displays the plot.
    title : str
        Title for the plot.
    """
    plt.figure(figsize=(8, 5))
    # Using plt.bar to represent the histogram, with width derived from bin_edges
    plt.bar(bin_edges[:-1], counts, width=np.diff(bin_edges), edgecolor="black", alpha=0.7)
    plt.title(title)
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Number of Pixels")
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()

    if outfile:
        Path(outfile).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outfile, dpi=300, bbox_inches="tight")
        plt.close() # Close the plot to free memory
    else:
        plt.show()
    return plt.gca() # Returns the current axes, similar to other plotting functions


def plot_tile_histograms_grid(
    all_tile_hist_data: List[Dict], # List of dicts, each containing 'tile_y', 'tile_x', 'counts', 'bin_edges'
    channel_name: str,
    outfile: Optional[Union[str, Path]] = None,
    title_prefix: str = "Pixel Intensity Histograms -",
    x_range: Tuple[float, float] = (0, 65535) # Default to uint16 range for consistency
):
    """
    Draws a grid of pixel intensity histograms for a single channel across multiple tiles.

    Parameters
    ----------
    all_tile_hist_data : List[Dict]
        A list of dictionaries, where each dictionary represents one tile's
        histogram data for the specified channel, containing keys:
        'tile_y', 'tile_x', 'counts', 'bin_edges'.
    channel_name : str
        The name of the channel being plotted.
    outfile : str or Path, optional
        Path to save the combined histogram plot. If None, displays the plot.
    title_prefix : str
        Prefix for the main plot title.
    x_range : tuple[float, float]
        The (min, max) range to use for the x-axis of all subplots, ensuring consistency.
    """
    if not all_tile_hist_data:
        warnings.warn("No histogram data provided for combined grid plot.")
        return None

    # Collect unique tile coordinates to determine actual grid span
    unique_ys = sorted(list(set(d['tile_y'] for d in all_tile_hist_data)))
    unique_xs = sorted(list(set(d['tile_x'] for d in all_tile_hist_data)))

    # Create a mapping from (y,x) coordinate to grid row/col index
    y_to_row_idx = {y: i for i, y in enumerate(unique_ys)}
    x_to_col_idx = {x: i for i, x in enumerate(unique_xs)}

    n_rows = len(unique_ys)
    n_cols = len(unique_xs)

    if n_rows == 0 or n_cols == 0:
        warnings.warn("Invalid grid dimensions based on tile coordinates. No plot generated.")
        return None

    # Adjust figure size dynamically based on number of subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * 3), squeeze=False) # squeeze=False ensures axes is always 2D
    fig.suptitle(f"{title_prefix} {channel_name}", fontsize=16)

    # Plot each histogram
    for hist_data in all_tile_hist_data:
        tile_y = hist_data['tile_y']
        tile_x = hist_data['tile_x']
        counts = hist_data['counts']
        bin_edges = hist_data['bin_edges']

        row_idx = y_to_row_idx[tile_y]
        col_idx = x_to_col_idx[tile_x]
        ax = axes[row_idx, col_idx]

        # Use plt.bar to represent the histogram
        ax.bar(bin_edges[:-1], counts, width=np.diff(bin_edges), edgecolor="black", alpha=0.7, align='edge')
        ax.set_title(f"Tile Y:{tile_y}, X:{tile_x}", fontsize=8)
        ax.set_xlim(x_range[0], x_range[1]) # Standardize x-axis range

        # Only label outermost axes to avoid clutter
        if row_idx == n_rows - 1: # Last row
            ax.set_xlabel("Intensity", fontsize=7)
        else:
            ax.set_xticklabels([])

        if col_idx == 0: # First column
            ax.set_ylabel("Count", fontsize=7)
        else:
            ax.set_yticklabels([])

        ax.tick_params(axis='both', which='major', labelsize=6)
        ax.grid(axis='y', alpha=0.75)

    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Leave space for suptitle

    if outfile:
        Path(outfile).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outfile, dpi=300, bbox_inches="tight")
        plt.close(fig) # Close the figure to free memory
    else:
        plt.show()

    return fig