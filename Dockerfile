######################################################################
# mIF-QC-Toolkit – production container
# ---------------------------------------------------------------
# • Based on the slim-Debian variant to minimise size
# • Installs the few system libraries that Matplotlib and PySAL
#   still need at run-time (freetype, libpng, libspatialindex)
# • Copies your source tree into /opt/mifqc and installs
#   it in “wheel” mode (editable install not needed in production)
# • Exposes the Typer CLI entry-point:  `mifqc`
######################################################################

# ---------- build stage -------------------------------------------------
FROM python:3.11-slim AS builder

# All Python wheels and pip cache live only in this stage
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential       \
        gcc g++               \
        libfreetype6-dev      \
        libpng-dev            \
        libspatialindex-dev   \
        pkg-config            \
    && rm -rf /var/lib/apt/lists/*

# Install your package plus runtime deps
WORKDIR /opt
COPY pyproject.toml README.md ./
COPY mifqc ./mifqc
RUN python -m pip install --upgrade pip wheel \
 && python -m pip install .                   \
 && python -m pip uninstall -y build          
 # trim unused build helper

# ---------- runtime stage ------------------------------------------------
FROM python:3.11-slim

# Keep only the shared libraries actually required at runtime
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libfreetype6 \
        libpng16-16  \
        libspatialindex6 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed site-packages from builder image
COPY --from=builder /usr/local /usr/local

# Default working directory
WORKDIR /workspace

# Display full version info at start-up (useful for bug reports)
ENTRYPOINT ["mifqc", "info"]
CMD        ["--help"]
