# mifqc/image_base.py
from __future__ import annotations
from pathlib import Path
import time
from tqdm import tqdm
import numpy as np
import pandas as pd
from tifffile import TiffFile,imread
from typing import Sequence, Tuple, List, Optional, Union
import warnings
import dask.array as da
import dask_image.imread as dair
import zarr
from ome_types import from_tiff
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from .qc_metrics import basic_stats, gini_index, geary_c, calculate_histogram

@dataclass
class ImageBase:
    """Abstract base for EntireImage and TiledImage."""
    pixels: np.ndarray                             # (C, H, W)
    channel_names: List[str]                       # length == C
    name: str = "unnamed"
    stats_: pd.DataFrame = field(init=False)
    _global_max_intensity: Optional[float] = field(init=False, default=None) # Cached attribute for global max

    # Property to calculate and cache global max intensity
    @property
    def global_max_intensity(self) -> float:
        if self._global_max_intensity is None:
            if self.pixels.size == 0:
                self._global_max_intensity = 0.0 # Handle empty image case
            else:
                # Check if it's a Dask array or NumPy array
                if isinstance(self.pixels, da.Array):
                    # Compute the max for Dask array
                    self._global_max_intensity = float(self.pixels.max().compute())
                else:
                    # Compute the max for NumPy array
                    self._global_max_intensity = float(self.pixels.max())
        return self._global_max_intensity

    # ---------- Constructors ----------
    @classmethod
    def from_tiff(cls, path: Union[str, Path], axes: Optional[str] = None,
              channel_names: Optional[List[str]] = None):
        with TiffFile(path) as tf:
            ser = tf.series[0]          # handle multi-series separately if needed
            axes_in = ser.axes if axes is None else axes
            arr = ser.asarray()         # honours memory-mapping or zarr if selected

            # Extract channel names INSIDE the context manager
            if channel_names is None:
                channel_names = _extract_channel_names(tf)

        if axes_in != "CYX":
            arr, axes_out = _canonicalize_axes(arr, axes_in, want="CYX")
            warnings.warn(f"Reordered axes {axes_in} → {axes_out}")
        return cls(arr, channel_names, name=Path(path).stem)

    # ---------- lazy TIFF/Zarr/Dask loading helpers ----------
    @classmethod
    def from_dask(cls, darr: da.Array, channel_names: Optional[List[str]] = None, name="unnamed"):
        if darr.ndim == 2:
            darr = darr[None, ...]
        if channel_names is None:
            channel_names = [f"ch{i}" for i in range(darr.shape[0])]
        return cls(darr, channel_names, name=name)

    @classmethod
    def from_zarr(cls, url: str, component: Optional[str] = None,
                  channel_names: Optional[List[str]] = None):
        darr = da.from_zarr(url, component=component)
        return cls.from_dask(darr, channel_names, name=Path(url).stem)

    @classmethod
    def from_glob(cls, pattern: str, channel_names: Optional[List[str]] = None):
        """Load N separate TIFFs (or PNGs) lazily as C×H×W via dask-image."""
        darr = dair.imread(pattern)      # shape = (N, H, W)
        return cls.from_dask(darr, channel_names, name=Path(pattern).stem)


    # ---------- Pixel-wise statistics ----------
    def per_channel_stats(self, show_progress: bool = True) -> pd.DataFrame:
        # progress bar and timings 
        start_time = time.time()

        # Create progress bar for channels
        if show_progress and len(self.channel_names) > 1:
            channel_iterator = tqdm(
                enumerate(self.channel_names),
                total=len(self.channel_names),
                desc=f"Analyzing {self.name} channels",
                unit="channels",
                leave=True
            )
        else:
            channel_iterator = enumerate(self.channel_names)

        rows: List[dict] = []
        for c, ch_name in enumerate(self.channel_names):
            band = self.pixels[c]

            # Update progress bar with current channel info
            if show_progress and len(self.channel_names) > 1 and hasattr(channel_iterator, 'set_postfix'):
                channel_iterator.set_postfix({
                    'current': ch_name,
                    'shape': f"{band.shape[0]}x{band.shape[1]}"
                })
            
            # Calculate statistics
            d = basic_stats(band)
            d["gini"] = gini_index(band)
            # skip geary_c for now
            #d["geary_c"] = geary_c(band)
            d["channel"] = ch_name
            rows.append(d)
        self.stats_ = pd.DataFrame(rows).set_index("channel")

        # Calculate and display timing
        elapsed_time = time.time() - start_time
        if show_progress:
            print(f"✓ [MIFQC] Analyzed {len(self.channel_names)} channels in {elapsed_time:.2f}s "
                f"({elapsed_time/len(self.channel_names):.3f}s per channel)")

        return self.stats_


    def per_channel_histograms(self, 
                               channels: Optional[Sequence[str]] = None, 
                               bins: int = 100, 
                               value_range: Optional[Tuple[float, float]] = None, 
                               show_progress: bool = True) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """
        Calculates pixel intensity histograms for each channel of the image.

        Parameters
        ----------
        channels : Sequence[str], optional
            List of channel names to calculate histograms for. If None, all channels are processed.
        bins : int
            Number of bins for the histogram.
        value_range : tuple[float, float]
            The (min, max) range for the histogram bins. If None, the range is
            determined by the min and max values of the array.
            Default is for uint16, 0 to 65535.
        show_progress : bool
            Whether to show a progress bar.

        Returns
        -------
        dict[str, tuple[np.ndarray, np.ndarray]]
            A dictionary where keys are channel names and values are (counts, bin_edges) tuples.
        """
        start_time = time.time()
        all_histograms = {}

        channels_to_process = list(self.channel_names) if channels is None else [ch for ch in channels if ch in self.channel_names]

        # Determine the actual value range for the histograms
        actual_value_range = value_range
        if actual_value_range is None:
            if self.pixels.size == 0:
                warnings.warn("Image has no pixels; cannot determine histogram range. Using (0,1).")
                actual_value_range = (0, 1) # Fallback for empty array
            else:
                # Use the new global_max_intensity property
                actual_value_range = (0, self.global_max_intensity)
                warnings.warn(f"Automatically set histogram range to (0, {actual_value_range[1]:.2f}) "
                              f"based on global max pixel value for this image/tile.")
        # reassign value_range after updating
        value_range = actual_value_range

        if show_progress and len(channels_to_process) > 1:
            channel_iterator = tqdm(
                channels_to_process,
                total=len(channels_to_process),
                desc=f"Calculating histograms for {self.name} channels",
                unit="channels",
                leave=True
            )
        else:
            channel_iterator = channels_to_process

        for ch_name in channel_iterator:
            c = self.channel_names.index(ch_name) # Get the index for the current channel name
            band = self.pixels[c]
            counts, bin_edges = calculate_histogram(band, bins=bins, value_range=value_range) # Pass value_range
            all_histograms[ch_name] = (counts, bin_edges)

            if show_progress and hasattr(channel_iterator, 'set_postfix'):
                channel_iterator.set_postfix({'current': ch_name})

        elapsed_time = time.time() - start_time
        if show_progress:
            print(f"✓ [MIFQC] Calculated histograms for {len(channels_to_process)} channels in {elapsed_time:.2f}s")
        return all_histograms


    # ---------- Export ----------
    def to_csv(self, out_path: Union[str, Path]) -> None:
        if not hasattr(self, "stats_"):
            self.per_channel_stats()
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        self.stats_.to_csv(out, index=True)
        print(f"[mif-qc] Wrote {out}")

# --- Helper to fix axes ordering if needed ---
def _canonicalize_axes(arr, axes: str, want: str = "CYX"):
    """
    Canonicalize axes by reordering array dimensions.
    Enhanced to handle Z->C conversion for multiplexed data.
    """
    
    # Detect and convert Z->C for multiplexed data
    if _should_convert_z_to_c(axes, arr):
        original_axes = axes
        axes = _convert_z_to_c_axes(axes)
        warnings.warn(f"[MIFQC] Detected multiplexed data: converting {original_axes} -> {axes} "
                     f"({arr.shape[0]} Z slices treated as channels)")
    
    # Validation with error messages
    axes_set = set(axes)
    want_set = set(want)
    
    if axes_set != want_set:
        missing_axes = want_set - axes_set
        extra_axes = axes_set - want_set
        error_msg = f"Unexpected axes string '{axes}'. Expected '{want}'"
        
        if missing_axes:
            error_msg += f". Missing axes: {missing_axes}"
        if extra_axes:
            error_msg += f". Extra axes: {extra_axes}"
            
        raise ValueError(error_msg)

    # Move each letter in `want` to the front in given order
    order: Sequence[int] = []
    for target_axis in want:
        source_idx = axes.index(target_axis)
        order.append(source_idx)
    
    # Transpose the array
    arr_reordered = arr.transpose(order)
    axes_out = "".join(want)
    
    return arr_reordered, axes_out


def _should_convert_z_to_c(axes: str, arr) -> bool:
    """
    Determine if Z axis should be treated as C axis.
    
    This detects multiplexed immunofluorescence data where channels
    are stored as Z slices instead of separate channels.
    
    Args:
        axes: Current axes string (e.g., "ZYX")
        arr: The image array
        
    Returns:
        bool: True if Z should be converted to C
    """
    # Must have Z axis but no C axis
    has_z = 'Z' in axes
    has_c = 'C' in axes
    
    if not has_z or has_c:
        return False
    
    # Get the Z dimension size
    z_axis_idx = axes.index('Z')
    z_size = arr.shape[z_axis_idx]
    
    # Convert if we have multiple Z slices that could be channels
    # Typical range for multiplexed immunofluorescence: 2-50 channels
    return 0 < z_size <= 500


def _convert_z_to_c_axes(axes: str) -> str:
    """
    Convert Z axis to C axis in the axes string.
    
    Args:
        axes: Original axes string (e.g., "ZYX")
        
    Returns:
        str: Modified axes string (e.g., "CYX")
    """
    return axes.replace('Z', 'C')



# --- Helper to extract tiff metadata ---
def _extract_channel_names(tf):
    """
    Enhanced channel name extraction that handles various formats.
    """
    
    # Strategy 1: Try OME-XML from ImageDescription tag
    try:
        # Get the first page's description which should contain OME-XML
        first_page = tf.pages[0]
        if hasattr(first_page, 'description') and first_page.description:
            ome_xml = first_page.description
            if ome_xml and '<OME' in ome_xml:
                # Parse the OME-XML to extract channel names
                root = ET.fromstring(ome_xml)
                
                # Define namespaces that might be used
                namespaces = {
                    'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06',
                    'ome2013': 'http://www.openmicroscopy.org/Schemas/OME/2013-06',
                    'ome2012': 'http://www.openmicroscopy.org/Schemas/OME/2012-06'
                }
                
                # Try different namespace combinations
                for ns_key, ns_url in namespaces.items():
                    try:
                        channels = root.findall(f'{{{ns_url}}}Image/{{{ns_url}}}Pixels/{{{ns_url}}}Channel')
                        if channels:
                            names = []
                            for i, ch in enumerate(channels):
                                name = ch.get('Name') or ch.get('name') or f"Channel_{i}"
                                names.append(name)
                            if names:
                                return names
                    except:
                        continue
                
                # Try without namespaces (sometimes XML doesn't have proper namespaces)
                try:
                    channels = root.findall('.//Channel')
                    if channels:
                        names = []
                        for i, ch in enumerate(channels):
                            name = ch.get('Name') or ch.get('name') or f"Channel_{i}"
                            names.append(name)
                        if names:
                            return names
                except:
                    pass
    except Exception as e:
        warnings.warn(f"Failed to extract from OME-XML: {e}")
    
    # Strategy 2: Try Lunaphore-specific channel information
    try:
        first_page = tf.pages[0]
        if hasattr(first_page, 'description') and first_page.description:
            ome_xml = first_page.description
            if ome_xml and 'FluorescenceChannel' in ome_xml:
                # Parse for Lunaphore-specific channel information
                root = ET.fromstring(ome_xml)
                # Look for ChannelPriv elements which Lunaphore uses
                channel_elements = []
                for elem in root.iter():
                    if elem.tag.endswith('ChannelPriv'):
                        channel_elements.append(elem)
                
                if channel_elements:
                    names = []
                    # Sort by channel ID to maintain order
                    channel_elements.sort(key=lambda x: x.get('ID', ''))
                    for elem in channel_elements:
                        fluor_channel = elem.get('FluorescenceChannel')
                        if fluor_channel:
                            names.append(fluor_channel)
                    if names:
                        return names
    except Exception as e:
        warnings.warn(f"Failed to extract Lunaphore channel info: {e}")
    
    # Strategy 3: Handle ImageJ format specifically
    try:
        if tf.imagej_metadata:
            ij = tf.imagej_metadata
            
            # Check for ImageJ Labels
            if "Labels" in ij and ij["Labels"]:
                labels = ij["Labels"].split("\n")
                if labels and all(labels):
                    return [label.strip() for label in labels if label.strip()]
            
            # For ImageJ format, use the number of slices/images
            if "slices" in ij and ij["slices"] > 1:
                num_channels = ij["slices"]
                return [f"Channel_{i:02d}" for i in range(num_channels)]
            elif "images" in ij and ij["images"] > 1:
                num_channels = ij["images"]
                return [f"Channel_{i:02d}" for i in range(num_channels)]
                
    except Exception as e:
        warnings.warn(f"Failed to extract from ImageJ metadata: {e}")
    
    # Strategy 4: Try PageName tags (safer approach for TiffFrame vs TiffPage issue)
    try:
        names = []
        for i, page in enumerate(tf.pages):
            try:
                # Handle both TiffPage and TiffFrame objects
                if hasattr(page, 'tags'):
                    # TiffPage object
                    if "PageName" in page.tags:
                        name = page.tags["PageName"].value
                        if name:
                            names.append(name)
                        else:
                            names.append(f"Channel_{i}")
                    else:
                        names.append(f"Channel_{i}")
                elif hasattr(page, 'keyframe') and hasattr(page.keyframe, 'tags'):
                    # TiffFrame object - access tags through keyframe
                    if "PageName" in page.keyframe.tags:
                        name = page.keyframe.tags["PageName"].value
                        if name:
                            names.append(name)
                        else:
                            names.append(f"Channel_{i}")
                    else:
                        names.append(f"Channel_{i}")
                else:
                    names.append(f"Channel_{i}")
            except Exception as e:
                warnings.warn(f"Error accessing page {i} tags: {e}")
                names.append(f"Channel_{i}")
        
        if names and any(name != f"Channel_{i}" for i, name in enumerate(names)):
            return names
    except Exception as e:
        warnings.warn(f"Failed to extract from PageName tags: {e}")
    
    # Strategy 5: Infer from axes and shape
    try:
        series = tf.series[0]
        axes = series.axes
        shape = series.shape
        
        # If we have ZYX or CYX format, use the first dimension
        if axes in ['ZYX', 'CYX'] and len(shape) >= 3:
            num_channels = shape[0]
            return [f"Channel_{i:02d}" for i in range(num_channels)]
        elif len(shape) == 2:
            return ["Channel_00"]  # Single channel 2D image
            
    except Exception as e:
        warnings.warn(f"Failed to determine channel count from axes: {e}")
    
    # Fallback
    return ["Channel_00"]
