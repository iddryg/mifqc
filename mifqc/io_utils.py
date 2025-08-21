# mifqc/io_utils.py
from pathlib import Path
from typing import Union
import pandas as pd
import numpy as np

def write_report(obj, out_dir: Union[str, Path]):
    """Dump whichever `.stats_` dataframe an object holds."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(obj, "stats_"):
        obj.stats_.to_csv(out_dir / f"{obj.name}_stats.csv")
    elif hasattr(obj, "_tile_stats"):
        obj._tile_stats.to_csv(out_dir / f"{obj.name}_tile_stats.csv", index=False)


def save_histogram_data(counts: np.ndarray, bin_edges: np.ndarray, out_path: Union[str, Path]):
    """
    Saves histogram counts and bin edges to a CSV file.

    Parameters
    ----------
    counts : np.ndarray
        The counts of pixels in each bin.
    bin_edges : np.ndarray
        The edges of the histogram bins.
    out_path : str or Path
        Path to save the CSV file.
    """
    df = pd.DataFrame({
        'bin_start': bin_edges[:-1],
        'bin_end': bin_edges[1:],
        'pixel_count': counts
    })
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[mif-qc] Wrote histogram data to {out_path}")