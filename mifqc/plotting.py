# mifqc/plotting.py
from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, Optional

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