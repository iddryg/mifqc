# mifqc/io_utils.py
from pathlib import Path
from typing import Union
import pandas as pd

def write_report(obj, out_dir: Union[str, Path]):
    """Dump whichever `.stats_` dataframe an object holds."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(obj, "stats_"):
        obj.stats_.to_csv(out_dir / f"{obj.name}_stats.csv")
    elif hasattr(obj, "_tile_stats"):
        obj._tile_stats.to_csv(out_dir / f"{obj.name}_tile_stats.csv", index=False)
