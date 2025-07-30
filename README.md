# MIFQC

Quality Control statistics for Mulitplex ImmunoFluorescence Images: (1) Entire Image, (2) Image Tiles, and (3) Cell Tables. 

Note: This project is still under development. 

A Python toolkit for **quality control (QC) metrics** in multiplex immunofluorescence (mIF) tissue imaging data.

The toolkit provides:

- Tools for image-wide, tiled, and cell-level QC metrics
- Support for large images and out-of-core processing using Dask and Zarr
- An intuitive command-line interface (CLI) for batch workflows
- Visualization utilities for rapid QC report generation (per-tile heatmaps)
- Ready-to-use Docker image and basic unit tests

---

## 📦 Project Description

**MIFQC** provides QC checks on whole-slide multiplex immunofluorescence images and their downstream single-cell feature tables.

Typical use cases include:

1. Checking image quality and uniformity across multiple fluorescence channels
2. Visualizing channel- or marker-level statistics (mean, Gini, Moran's I, etc.) at global, tiled, or cell level resolution
3. Assessing spatial structure and tissue coverage for each channel
4. Streamlining high-throughput imaging studies via CLI, Python API, or containerized workflows

---

## 🚀 Installation

### 1. From source (pip)

```bash
# clone or download the repo, then:
pip install -e .
```

#### System libraries (Debian/Ubuntu)

Some compiled wheels require run-time libraries:

```bash
sudo apt-get update
sudo apt-get install -y libfreetype6 libpng16-16 libspatialindex6
```

### 2. Using Docker

1. Build the image:

   ```bash
   docker build -t mifqc .
   ```

2. Verify installed versions:

   ```bash
   docker run --rm mifqc
   ```

---

## 🔥 Quick-Start Usage

### 1. Python API

```python
from mifqc import (
    EntireImage, TiledImage, CellTable,
    quick_whole_image_qc, quick_tile_analysis,
)

# Whole-image QC
img = EntireImage.from_tiff("slideA.ome.tiff", channel_names=["DAPI", "CD3", "CD8"])
img.per_channel_stats()
img.to_csv("qc/slideA_whole_image_stats.csv")

# Tile-based QC + visualization
tiles = TiledImage(img.pixels, img.channel_names, name=img.name, tile_size=512)
tiles.tile_statistics()
tiles.tiles_to_csv("qc/")

# Convenience wrappers
quick_whole_image_qc(
    "slideA.ome.tiff", channel_names=["DAPI", "CD3", "CD8"], output_dir="qc"
)
quick_tile_analysis(
    "slideA.ome.tiff", tile_size=512, channel_names=["DAPI", "CD3", "CD8"], output_dir="qc"
)
```

### 2. Command-Line Interface

| Action | Example |
|--------|---------|
| Show package info | `mifqc info` |
| Whole-image QC | `mifqc whole slideA.ome.tiff --out qc/` |
| Tile QC + heatmap | `mifqc tile slideA.ome.tiff --channel CD3 --metric moran_I --tile-size 512 --out qc/` |

### 3. Docker-based Workflows

```bash
docker run --rm \
  -v $PWD/data:/data \
  -v $PWD/qc_out:/out \
  mifqc tile /data/slideA.ome.tiff --channel CD3 --out /out
```

---

## 💡 Tips & Troubleshooting

* Use Zarr + Dask for slides too large for RAM: `EntireImage.from_zarr()` or the `mifqc` CLI works lazily.
* CSV and PNG outputs are auto-named inside your `--out` directory.
* Run tests with `pytest -n auto --cov=mifqc`.

---
>>>>>>> d8d2557 (Initial commit)

# mIF-QC-Toolkit

Note: This project is still under development. 

A Python toolkit for **quality control (QC) metrics** in multiplex immunofluorescence (mIF) tissue imaging data.

The toolkit provides:

- Tools for image-wide, tiled, and cell-level QC metrics
- Support for large images and out-of-core processing using Dask and Zarr
- An intuitive command-line interface (CLI) for batch workflows
- Visualization utilities for rapid QC report generation (per-tile heatmaps)
- Ready-to-use Docker image and basic unit tests

---

## 📦 Project Description

**MIFQC** provides QC checks on whole-slide multiplex immunofluorescence images and their downstream single-cell feature tables.

Typical use cases include:

1. Checking image quality and uniformity across multiple fluorescence channels
2. Visualizing channel- or marker-level statistics (mean, Gini, Moran's I, etc.) at global, tiled, or cell level resolution
3. Assessing spatial structure and tissue coverage for each channel
4. Streamlining high-throughput imaging studies via CLI, Python API, or containerized workflows

---

## 🚀 Installation

### 1. From source (pip)

```bash
# clone or download the repo, then:
pip install -e .
```

#### System libraries (Debian/Ubuntu)

Some compiled wheels require run-time libraries:

```bash
sudo apt-get update
sudo apt-get install -y libfreetype6 libpng16-16 libspatialindex6
```

### 2. Using Docker

1. Build the image:

   ```bash
   docker build -t mifqc .
   ```

2. Verify installed versions:

   ```bash
   docker run --rm mifqc
   ```

---

## 🔥 Quick-Start Usage

### 1. Python API

```python
from mifqc import (
    EntireImage, TiledImage, CellTable,
    quick_whole_image_qc, quick_tile_analysis,
)

# Whole-image QC
img = EntireImage.from_tiff("slideA.ome.tiff", channel_names=["DAPI", "CD3", "CD8"])
img.per_channel_stats()
img.to_csv("qc/slideA_whole_image_stats.csv")

# Tile-based QC + visualization
tiles = TiledImage(img.pixels, img.channel_names, name=img.name, tile_size=512)
tiles.tile_statistics()
tiles.tiles_to_csv("qc/")

# Convenience wrappers
quick_whole_image_qc(
    "slideA.ome.tiff", channel_names=["DAPI", "CD3", "CD8"], output_dir="qc"
)
quick_tile_analysis(
    "slideA.ome.tiff", tile_size=512, channel_names=["DAPI", "CD3", "CD8"], output_dir="qc"
)
```

### 2. Command-Line Interface

| Action | Example |
|--------|---------|
| Show package info | `mifqc info` |
| Whole-image QC | `mifqc whole slideA.ome.tiff --out qc/` |
| Tile QC + heatmap | `mifqc tile slideA.ome.tiff --channel CD3 --metric moran_I --tile-size 512 --out qc/` |

### 3. Docker-based Workflows

```bash
docker run --rm \
  -v $PWD/data:/data \
  -v $PWD/qc_out:/out \
  mifqc tile /data/slideA.ome.tiff --channel CD3 --out /out
```

---

## 💡 Tips & Troubleshooting

* Use Zarr + Dask for slides too large for RAM: `EntireImage.from_zarr()` or the `mifqc` CLI works lazily.
* CSV and PNG outputs are auto-named inside your `--out` directory.
* Run tests with `pytest -n auto --cov=mifqc`.

---
