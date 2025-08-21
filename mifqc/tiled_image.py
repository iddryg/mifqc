# mifqc/tiled_image.py
from __future__ import annotations
import time
from tqdm import tqdm
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Union
from .entire_image import EntireImage
from .plotting import heatmap, plot_histogram, plot_tile_histograms_grid
from .io_utils import write_report, save_histogram_data, save_combined_histogram_data
import warnings

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
                        histogram_bins: int = 100,
                        standardized_histogram_range: Optional[Tuple[float, float]] = None, # assume uint16
                        output_base_dir: Optional[Path] = None, # Base directory for all outputs
                        histogram_output_subdir: str = "histograms") -> pd.DataFrame:
        """
        Computes per-tile statistics and optionally generates/saves histograms.

        Parameters
        ----------
        max_tiles : Optional[int]
            Maximum number of tiles to process.
        show_progress : bool
            Whether to show a progress bar.
        save_histograms : bool
            If True, calculates, plots, and saves combined histogram data for each channel.
        histogram_bins : int
            Number of bins to use for histograms if `save_histograms` is True.
        standardized_histogram_range : tuple[float, float], optional
            If provided, fixes the (min, max) range for all histograms.
            e.g., (0, 65535) for uint16 images.
        output_base_dir : Path, optional
            The root directory where all outputs (including histograms) should be saved.
            If None, a default like "qc" will be used.
        histogram_output_subdir : str
            Subdirectory name (relative to output_base_dir) to save histogram plots and data.

        Returns
        -------
        pd.DataFrame
            DataFrame with per-tile statistics.
        """

        # progress bar and timing
        start_time = time.time()

        # determine base output directory
        if output_base_dir is None:
            output_base_dir = Path("qc") # Default if not provided
        output_base_dir = Path(output_base_dir)
        output_base_dir.mkdir(parents=True, exist_ok=True)

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
        # Store histogram data for plotting (grouped by channel, then list of tile data)
        # e.g., {'DAPI': [{'tile_y': Y, 'tile_x': X, 'counts': counts, 'bin_edges': edges}, ...]}
        all_channel_histogram_data_for_plotting: Dict[str, List[Dict[str, Union[int, np.ndarray]]]] = {}

        # Store histogram data for combined CSV (list of per-bin records)
        # e.g., [{'tile_y': Y, 'tile_x': X, 'tile_id': ID, 'channel': CH, 'bin_start': S, 'bin_end': E, 'pixel_count': P}, ...]
        combined_csv_records: List[Dict] = []

        for idx, (coord, tarr) in tile_iterator:
            tile_name_id = f"tile_{coord[0]}_{coord[1]}" # Unique identifier for tile files
            img = EntireImage(tarr, self.channel_names, name=f"{self.name}_tile{idx}")

            # Calculate scalar statistics
            stats = img.per_channel_stats(show_progress=False) # Don't show nested progress
            stats = stats.reset_index() # get channel names as a column
            stats["tile_y"], stats["tile_x"] = coord
            stats["tile_id"] = idx
            rows.append(stats)

            # Calculate Histograms per Tile and collect for combined output
            if save_histograms:
                # Calculate histograms for all channels in the current tile
                tile_histograms = img.per_channel_histograms(
                    channels=self.channel_names, # Process all channels for this tile
                    bins=histogram_bins,
                    value_range=standardized_histogram_range, # Pass new parameter
                    show_progress=False
                )

                for ch_name, (counts, bin_edges) in tile_histograms.items():
                    # For combined plotting (store per-tile histogram data)
                    if ch_name not in all_channel_histogram_data_for_plotting:
                        all_channel_histogram_data_for_plotting[ch_name] = []
                    all_channel_histogram_data_for_plotting[ch_name].append({
                        'tile_y': coord[0],
                        'tile_x': coord[1],
                        'counts': counts,
                        'bin_edges': bin_edges
                    })

                    # For combined CSV (store per-bin details)
                    for i in range(len(counts)):
                        combined_csv_records.append({
                            'tile_y': coord[0],
                            'tile_x': coord[1],
                            'tile_id': idx,
                            'channel': ch_name,
                            'bin_start': bin_edges[i],
                            'bin_end': bin_edges[i+1],
                            'pixel_count': counts[i]
                        })

            # Update progress bar with current tile info
            if show_progress and hasattr(tile_iterator, 'set_postfix'):
                tile_iterator.set_postfix({
                    'current_tile': f"({coord[0]},{coord[1]})",
                    'channels': len(self.channel_names)
                })
        
        self._tile_stats = pd.concat(rows, ignore_index=True)
        
        # AFTER ALL TILES ARE PROCESSED: Plot and Save Combined Histograms
        if save_histograms:
            combined_hist_output_dir = output_base_dir / self.name / histogram_output_subdir
            combined_hist_output_dir.mkdir(parents=True, exist_ok=True)

            # 1. Save Combined Histogram CSV
            save_combined_histogram_data(
                combined_csv_records,
                combined_hist_output_dir / f"{self.name}_combined_tile_histograms_data.csv"
            )

            # 2. Generate Combined Histogram Plots (one per channel)
            for ch_name, hist_data_list_for_plotting in all_channel_histogram_data_for_plotting.items():
                plot_tile_histograms_grid(
                    hist_data_list_for_plotting,
                    channel_name=ch_name,
                    outfile=combined_hist_output_dir / f"{self.name}_{ch_name}_combined_tile_histograms.png",
                    x_range=standardized_histogram_range if standardized_histogram_range else (0, 65535) # Pass range for plotting
                )

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
