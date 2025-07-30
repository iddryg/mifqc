# mifqc/tiled_image.py
from __future__ import annotations
import numpy as np
import pandas as pd
from itertools import product
from dataclasses import dataclass, field
from .entire_image import EntireImage

@dataclass
class TiledImage(EntireImage):
    tile_size: int = 512
    stride: int | None = None
    _tile_stats: pd.DataFrame = field(init=False)

    def _iter_tiles(self):
        H, W = self.pixels.shape[1:]
        step = self.stride or self.tile_size
        for y, x in product(range(0, H - self.tile_size + 1, step),
                            range(0, W - self.tile_size + 1, step)):
            sl = np.s_[..., y : y + self.tile_size, x : x + self.tile_size]
            yield (y, x), self.pixels[sl]

    # ---------- Tile statistics ----------
    def tile_statistics(self, max_tiles: int | None = None) -> pd.DataFrame:
        rows = []
        for idx, (coord, tarr) in enumerate(self._iter_tiles()):
            if max_tiles and idx >= max_tiles:
                break
            img = EntireImage(tarr, self.channel_names, name=f"{self.name}_tile{idx}")
            stats = img.per_channel_stats()
            stats["tile_y"], stats["tile_x"] = coord
            stats["tile_id"] = idx
            rows.append(stats)
        self._tile_stats = pd.concat(rows)
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
