# mifqc/cli.py
from __future__ import annotations
import typer, sys
from pathlib import Path
from .entire_image import EntireImage
from .tiled_image  import TiledImage
from .plotting     import heatmap

app = typer.Typer(add_completion=False)

@app.command(help="Whole-image QC on an OME-TIFF, PNG glob or Zarr store.")
def whole(
    src        : str = typer.Argument(..., help="Path / glob / Zarr URL"),
    out        : Path = typer.Option("qc", help="Output folder"),
    zarr_comp  : str | None = typer.Option(None, "--component", "-c",
                                           help="Zarr sub-array name"),
):
    if src.endswith(".zarr") or "::" in src:
        img = EntireImage.from_zarr(src, component=zarr_comp)
    elif "*" in src:
        img = EntireImage.from_glob(src)
    else:
        img = EntireImage.from_tiff(src)

    df = img.per_channel_stats()
    out.mkdir(exist_ok=True, parents=True)
    df.to_csv(out / f"{img.name}_whole.csv")
    typer.echo(f"✓ whole-image stats → {out}")

@app.command(help="Per-tile QC and heat-map.")
def tile(
    src       : str = typer.Argument(..., help="Path/glob/Zarr"),
    out       : Path = typer.Option("qc", help="Output folder"),
    metric    : str = typer.Option("mean", help="Metric for heat-map"),
    channel   : str = typer.Option("DAPI", help="Channel for heat-map"),
    tile_size : int = typer.Option(512, help="Tile side length"),
):
    if src.endswith(".zarr") or "::" in src:
        base = EntireImage.from_zarr(src)
    elif "*" in src:
        base = EntireImage.from_glob(src)
    else:
        base = EntireImage.from_tiff(src)

    tiled = TiledImage(base.pixels, base.channel_names,
                       name=base.name, tile_size=tile_size)
    df = tiled.tile_statistics()
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{base.name}_per_tile.csv", index=False)
    heatmap(df, metric=metric, channel=channel,
            outfile=out / f"{base.name}_{channel}_{metric}.png")
    typer.echo(f"✓ tile stats + heat-map → {out}")

@app.command(help="Print library versions for bug reports.")
def info():
    import numpy, dask, tifffile, zarr, pysal, matplotlib
    mod = dict(numpy=numpy, dask=dask, tifffile=tifffile,
               zarr=zarr, pysal=pysal, matplotlib=matplotlib)
    for k, m in mod.items():
        typer.echo(f"{k:10s}: {m.__version__}")

if __name__ == "__main__":
    sys.exit(app())
