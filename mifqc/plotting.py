# mifqc/plotting.py
from __future__ import annotations
import matplotlib.pyplot as plt
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
