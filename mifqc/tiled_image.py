# mifqc/tiled_image.py
from __future__ import annotations
import time
from tqdm import tqdm
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product
from dataclasses import dataclass, field
from typing import Optional
from .entire_image import EntireImage
from .plotting import heatmap, plot_histogram
from .io_utils import write_report, save_histogram_data

@dataclass
class TiledImage(EntireImage):
    tile_size: int = 512
    stride: Optional[int] = None
    _tile_stats: pd.DataFrame = field(init=False)

    def _iter_tiles(self):
        H, W = self.pixels.shape[1:]
        step = self.stride or self.tile_size
        for y, x in product(range(0, H - self.tile_size + 1, step),
                            range(0, W - self.tile_size + 1, step)):
            sl = np.s_[..., y : y + self.tile_size, x : x + self.tile_size]
            yield (y, x), self.pixels[sl]

    # ---------- Tile statistics ----------
    def tile_statistics(self, 
                        max_tiles: Optional[int] = None, 
                        show_progress: bool = True,
                        save_histograms: bool = True, 
                        histogram_bins: int = 256,
                        histogram_output_subdir: str = "histograms") -> pd.DataFrame:
        """
        Computes per-tile statistics and optionally generates/saves histograms.

        Parameters
        ----------
        
        save_histograms : bool
            If True, calculates, plots, and saves histogram data for each channel in each tile.
        histogram_bins : int
            Number of bins to use for histograms if `save_histograms` is True.
        histogram_output_subdir : str
            Subdirectory within the output directory to save histogram plots and data.

        Returns
        -------
        pd.DataFrame
            DataFrame with per-tile statistics.
        """

        # progress bar and timing
        start_time = time.time()
        # Count total tiles for progress bar
        tiles_list = list(self._iter_tiles())
        if max_tiles:
            tiles_list = tiles_list[:max_tiles]
        total_tiles = len(tiles_list)

        # Create progress bar iterator
        if show_progress:
            tile_iterator = tqdm(
                enumerate(tiles_list), 
                total=total_tiles,
                desc=f"Processing {self.name} tiles",
                unit="tiles",
                leave=True
            )
        else:
            tile_iterator = enumerate(tiles_list)

        rows = []
        for idx, (coord, tarr) in tile_iterator:
            tile_name_id = f"tile_{coord}_{coord[7]}" # Unique identifier for tile files
            img = EntireImage(tarr, self.channel_names, name=f"{self.name}_tile{idx}")

            # Calculate scalar statistics
            stats = img.per_channel_stats(show_progress=False)  # Don't show nested progress
            stats = stats.reset_index() # get channel names as a column
            stats["tile_y"], stats["tile_x"] = coord
            stats["tile_id"] = idx
            rows.append(stats)
            
            # Calculate, Save, and Plot Histograms per Tile
            if save_histograms:
                # Create a specific output directory for this tile's histograms
                # Using Path(self.name).parent here assumes self.name is a path, which it might not be.
                # A more robust way would be to pass the base output directory from the caller.
                # For this example, let's assume `qc` is the top-level output and create sub-folders.
                base_output_dir = Path("qc") # Placeholder for the root QC output directory
                tile_histogram_output_path = base_output_dir / self.name / tile_name_id / histogram_output_subdir
                tile_histogram_output_path.mkdir(parents=True, exist_ok=True)

                tile_histograms = img.per_channel_histograms(bins=histogram_bins, show_progress=False)
                for ch_name, (counts, bin_edges) in tile_histograms.items():
                    # Save histogram data
                    save_histogram_data(
                        counts, bin_edges,
                        tile_histogram_output_path / f"{self.name}_{tile_name_id}_{ch_name}_histogram_data.csv"
                    )
                    # Plot and save histogram
                    plot_histogram(
                        counts, bin_edges,
                        outfile=tile_histogram_output_path / f"{self.name}_{tile_name_id}_{ch_name}_histogram.png",
                        title=f"Intensity Histogram - {ch_name} (Tile Y:{coord}, X:{coord[7]})"
                    )

            # Update progress bar with current tile info
            if show_progress and hasattr(tile_iterator, 'set_postfix'):
                tile_iterator.set_postfix({
                    'current_tile': f"({coord[0]},{coord[1]})",
                    'channels': len(self.channel_names)
                })
        
        self._tile_stats = pd.concat(rows, ignore_index=True)
        
        # Calculate and display timing
        elapsed_time = time.time() - start_time
        if show_progress:
            print(f"✓ [MIFQC] Processed {total_tiles} tiles in {elapsed_time:.2f}s "
                f"({elapsed_time/total_tiles:.3f}s per tile)")
        
        return self._tile_stats

    # ---------- Aggregate across tiles ----------
    def summarize_tiles(self) -> pd.DataFrame:
        if not hasattr(self, "_tile_stats"):
            self.tile_statistics()
        grouped = self._tile_stats.groupby("channel").agg(["mean", "std"])
        # Flatten multi-index columns
        grouped.columns = ["_".join(c) for c in grouped.columns]
        return grouped

    # ---------- Export ----------
    def tiles_to_csv(self, folder: str):
        if not hasattr(self, "_tile_stats"):
            self.tile_statistics()
        out = Path(folder) / f"{self.name}_per_tile.csv"
        self._tile_stats.to_csv(out, index=False)
        print(f"[mif-qc] wrote {out}")
