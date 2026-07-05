import os
import tempfile
from pathlib import Path
import numpy as np
import rasterio
from rasterio.errors import RasterioIOError
from rasterio.merge import merge
from rasterio.warp import reproject, Resampling
from pyproj import Transformer
from dem_stitcher import stitch_dem
import matplotlib.pyplot as plt


def _collect_readable_tifs(files):
    srcs = []
    skipped = []
    for tile in files:
        if not tile.exists():
            skipped.append(str(tile))
            continue
        try:
            src = rasterio.open(tile)
            try:
                for _, window in src.block_windows(1):
                    src.read(1, window=window)
            except RasterioIOError:
                skipped.append(str(tile))
                src.close()
                continue
            srcs.append(src)
        except RasterioIOError:
            skipped.append(str(tile))
    return srcs, skipped


def _write_merged_tif(srcs, output_tif, compress="deflate", quantize_step=None):
    output_tif = Path(output_tif)
    output_tif.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        suffix=".tif",
        prefix=f"{output_tif.stem}_merge_",
        dir=output_tif.parent,
        delete=False,
    ) as tmp_file:
        temp_tif = Path(tmp_file.name)

    merge_profile = {
        "driver": "GTiff",
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "compress": compress if compress else "zstd",
        "bigtiff": "IF_SAFER",
    }

    try:
        try:
            merge(srcs, dst_path=temp_tif, dst_kwds=merge_profile)
        finally:
            for src in srcs:
                src.close()

        with rasterio.open(temp_tif) as merged_src:
            profile = merged_src.profile.copy()
            src_nodata = merged_src.nodata

            if quantize_step is None:
                profile.update(dtype="float32", predictor=3)
                with rasterio.open(output_tif, "w", **profile) as dst:
                    for _, window in merged_src.block_windows(1):
                        data = merged_src.read(1, window=window).astype(np.float32, copy=False)
                        dst.write(data, 1, window=window)
                return output_tif

            if quantize_step <= 0:
                raise ValueError("quantize_step must be > 0")

            min_value = None
            max_value = None
            for _, window in merged_src.block_windows(1):
                band = merged_src.read(1, window=window)
                valid = np.isfinite(band)
                if src_nodata is not None:
                    valid &= band != src_nodata
                if not np.any(valid):
                    continue
                window_values = band[valid]
                window_min = float(window_values.min())
                window_max = float(window_values.max())
                min_value = window_min if min_value is None else min(min_value, window_min)
                max_value = window_max if max_value is None else max(max_value, window_max)

            if min_value is None:
                raise ValueError("No valid DEM pixels found for quantization")

            offset = float(np.floor(min_value))
            max_scaled = np.round((max_value - offset) / float(quantize_step))
            if max_scaled > 32767:
                raise ValueError("quantize_step too small for int16 range; increase quantize_step")

            profile.update(dtype="int16", nodata=-32768, predictor=2)
            with rasterio.open(output_tif, "w", **profile) as dst:
                for _, window in merged_src.block_windows(1):
                    band = merged_src.read(1, window=window)
                    valid = np.isfinite(band)
                    if src_nodata is not None:
                        valid &= band != src_nodata
                    scaled = np.full(band.shape, -32768, dtype=np.int16)
                    if np.any(valid):
                        values = np.round((band[valid] - offset) / float(quantize_step))
                        scaled[valid] = values.astype(np.int16)
                    dst.write(scaled, 1, window=window)
                dst.scales = [float(quantize_step)]
                dst.offsets = [offset]
            return output_tif
    finally:
        try:
            if temp_tif.exists():
                temp_tif.unlink()
        except Exception:
            pass

def save_reprojected_dem(elevation_data, profile, region, output_path, resolution=30.0):
    """Reprojects and saves raw downloaded DEM data to match the Region object's UTM grid."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Calculate target image dimensions and origin transform matrix
    width = int(np.ceil((region.east_m - region.west_m) / resolution))
    height = int(np.ceil((region.north_m - region.south_m) / resolution))
    transform = rasterio.transform.from_origin(region.west_m, region.north_m, resolution, resolution)
    
    meta = {
        'driver': 'GTiff', 'height': height, 'width': width,
        'count': 1, 'dtype': 'float32', 'crs': region.epsg, 'transform': transform, 'nodata': np.nan
    }
    
    # Reproject and write to disk
    with rasterio.open(output_path, 'w', **meta) as dst:
        reproject(
            source=elevation_data, destination=rasterio.band(dst, 1),
            src_transform=profile['transform'], src_crs=profile['crs'],
            dst_transform=transform, dst_crs=region.epsg,
            src_nodata=profile.get('nodata', np.nan),
            dst_nodata=np.nan,
            resampling=Resampling.bilinear
        )
        
    print(f"✓ Saved reprojected UTM DEM to: {output_path}")


def save_reprojected_dem_from_file(source_path, region, output_path, resolution=30.0):
    """Reprojects a DEM raster file to the Region object's UTM grid without loading it fully into memory."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    width = int(np.ceil((region.east_m - region.west_m) / resolution))
    height = int(np.ceil((region.north_m - region.south_m) / resolution))
    transform = rasterio.transform.from_origin(region.west_m, region.north_m, resolution, resolution)

    meta = {
        'driver': 'GTiff',
        'height': height,
        'width': width,
        'count': 1,
        'dtype': 'float32',
        'crs': region.epsg,
        'transform': transform,
        'nodata': np.nan,
        'tiled': True,
        'blockxsize': 512,
        'blockysize': 512,
        'compress': 'deflate',
        'predictor': 3,
        'bigtiff': 'IF_SAFER',
    }

    with rasterio.open(source_path) as src:
        with rasterio.open(output_path, 'w', **meta) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=region.epsg,
                src_nodata=src.nodata,
                dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )

    print(f"✓ Saved reprojected UTM DEM to: {output_path}")


def plot_dem(region, dem_path):
    """Loads the DEM from disk, prints simple statistics, and renders the plot with decimation."""
    import numpy as np
    import rasterio
    import matplotlib.pyplot as plt

    with rasterio.open(dem_path) as src:
        bounds = src.bounds
        nodata_val = src.nodata
        is_int16 = src.dtypes[0] == 'int16'
        scale_factor = max(1, src.width // 1000, src.height // 1000)
        target_height = src.height // scale_factor
        target_width = src.width // scale_factor
        data = src.read(1, out_shape=(1, target_height, target_width), resampling=Resampling.bilinear)
        
        # Handle quantized int16 with scale/offset BEFORE masking
        if is_int16:
            scales = getattr(src, 'scales', [1.0])
            offsets = getattr(src, 'offsets', [0.0])
            scale = scales[0] if scales else 1.0
            offset = offsets[0] if offsets else 0.0
            # Mask nodata BEFORE applying scale/offset
            if nodata_val is not None:
                data = np.where(data == nodata_val, np.nan, data.astype(np.float32))
            data = data * scale + offset
    
    if nodata_val is not None and np.isfinite(nodata_val) and not is_int16:
        data = np.where(data == nodata_val, np.nan, data)
        
    print(f"DEM Statistics:")
    print(f" -> Minimum Elevation: {np.nanmin(data):.1f} meters")
    print(f" -> Maximum Elevation: {np.nanmax(data):.1f} meters")
    print(f" -> Missing Data (NaN): {(np.isnan(data).sum() / data.size) * 100.0:.2f}%\n")

    data = np.ma.masked_invalid(data)
    extent_km = [bounds.left / 1000.0, bounds.right / 1000.0, bounds.bottom / 1000.0, bounds.top / 1000.0]
    
    plt.figure(figsize=(10, 8))
    cmap = plt.get_cmap('terrain').copy()
    cmap.set_bad(alpha=0.0)
    img = plt.imshow(data, cmap=cmap, extent=extent_km)
    plt.colorbar(img, label="Elevation (meters)")
    plt.title("Region Elevation Map", fontsize=13, fontweight='bold')
    plt.xlabel("Easting (km)")
    plt.ylabel("Northing (km)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()


def merge_tiles_to_compressed_tif(
    input_dir,
    output_tif,
    pattern="*.tif",
    compress="deflate",
    quantize_step=None,
):
    """Merges DEM tiles into one compressed GeoTIFF.

    If quantize_step is provided (meters), output is int16 with scale/offset tags.
    """
    input_dir = Path(input_dir)
    output_tif = Path(output_tif)
    tiles = sorted(input_dir.rglob(pattern))
    if not tiles:
        raise FileNotFoundError(f"No files matching {pattern} under {input_dir}")

    srcs, skipped = _collect_readable_tifs(tiles)

    if not srcs:
        raise ValueError(f"No readable GeoTIFF tiles found under {input_dir}")

    if skipped:
        print(f"Skipped {len(skipped)} unreadable tile(s):")
        for skipped_tile in skipped:
            print(f" - {skipped_tile}")
    return _write_merged_tif(srcs, output_tif, compress=compress, quantize_step=quantize_step)


def create_hybrid_dem_blockwise(
    glo30_path,
    hma_path,
    output_path,
    hma_max_elevation=8900.0,
    max_abs_diff=600.0,
):
    """Creates a hybrid DEM on the GLO-30 grid using HMA where valid, blockwise.

    This avoids allocating full-size arrays in memory.
    """
    glo30_path = Path(glo30_path)
    hma_path = Path(hma_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_pixels = 0
    hma_valid_pixels = 0
    hma_used_pixels = 0
    rejected_by_diff_pixels = 0
    rejected_by_ceiling_pixels = 0

    with rasterio.open(glo30_path) as src_glo, rasterio.open(hma_path) as src_hma:
        profile = src_glo.profile.copy()
        profile.update(
            dtype="float32",
            nodata=np.nan,
            compress="deflate",
            predictor=3,
            tiled=True,
            blockxsize=512,
            blockysize=512,
            bigtiff="IF_SAFER",
        )

        hma_nodata = src_hma.nodata
        hma_is_int16 = src_hma.dtypes[0] == "int16"
        hma_scale = (src_hma.scales[0] if getattr(src_hma, "scales", None) else 1.0) or 1.0
        hma_offset = (src_hma.offsets[0] if getattr(src_hma, "offsets", None) else 0.0) or 0.0

        with rasterio.open(output_path, "w", **profile) as dst:
            for _, window in dst.block_windows(1):
                glo = src_glo.read(1, window=window).astype(np.float32, copy=False)
                glo_nodata = src_glo.nodata
                if glo_nodata is not None and np.isfinite(glo_nodata):
                    glo = np.where(glo == glo_nodata, np.nan, glo)

                hma_reproj = np.full((window.height, window.width), np.nan, dtype=np.float32)
                reproject(
                    source=rasterio.band(src_hma, 1),
                    destination=hma_reproj,
                    src_transform=src_hma.transform,
                    src_crs=src_hma.crs,
                    dst_transform=rasterio.windows.transform(window, dst.transform),
                    dst_crs=dst.crs,
                    src_nodata=hma_nodata,
                    dst_nodata=np.nan,
                    resampling=Resampling.bilinear,
                )

                if hma_is_int16:
                    hma_reproj = hma_reproj * float(hma_scale) + float(hma_offset)

                too_high = np.isfinite(hma_reproj) & (hma_reproj > float(hma_max_elevation))
                rejected_by_ceiling_pixels += int(np.count_nonzero(too_high))
                hma_reproj = np.where(too_high, np.nan, hma_reproj)

                valid_hma = np.isfinite(hma_reproj)
                hma_valid_pixels += int(np.count_nonzero(valid_hma))

                if max_abs_diff is not None:
                    comparable = valid_hma & np.isfinite(glo)
                    too_diff = comparable & (np.abs(hma_reproj - glo) > float(max_abs_diff))
                    rejected_by_diff_pixels += int(np.count_nonzero(too_diff))
                    valid_hma = valid_hma & ~too_diff

                # Conservative fusion: only trusted HMA pixels can override, and peaks are preserved.
                hma_candidate = np.where(valid_hma, hma_reproj, np.nan)
                hybrid = np.fmax(glo, hma_candidate)
                hma_used = valid_hma & (np.isnan(glo) | (hma_candidate > glo))
                hma_used_pixels += int(np.count_nonzero(hma_used))
                total_pixels += int(window.height * window.width)
                dst.write(hybrid.astype(np.float32, copy=False), 1, window=window)

    stats = {
        "total_pixels": int(total_pixels),
        "hma_valid_pixels": int(hma_valid_pixels),
        "hma_used_pixels": int(hma_used_pixels),
        "hma_used_pct": (100.0 * hma_used_pixels / total_pixels) if total_pixels else 0.0,
        "rejected_by_diff_pixels": int(rejected_by_diff_pixels),
        "rejected_by_ceiling_pixels": int(rejected_by_ceiling_pixels),
        "max_abs_diff": None if max_abs_diff is None else float(max_abs_diff),
        "hma_max_elevation": float(hma_max_elevation),
    }
    print(f"✓ Saved hybrid DEM to: {output_path}")
    print(
        "Hybrid stats -> "
        f"HMA used: {stats['hma_used_pixels']:,}/{stats['total_pixels']:,} "
        f"({stats['hma_used_pct']:.2f}%), "
        f"rejected diff: {stats['rejected_by_diff_pixels']:,}, "
        f"rejected ceiling: {stats['rejected_by_ceiling_pixels']:,}"
    )
    return stats


def merge_geotiffs_to_compressed_tif(
    input_files,
    output_tif,
    compress="deflate",
    quantize_step=None,
):
    """Merges a list of GeoTIFF files into one compressed GeoTIFF.

    If quantize_step is provided (meters), output is int16 with scale/offset tags.
    """
    output_tif = Path(output_tif)
    files = [Path(p) for p in input_files]
    if not files:
        raise FileNotFoundError("No input GeoTIFF files provided")

    srcs, skipped = _collect_readable_tifs(files)

    if not srcs:
        raise ValueError("No readable GeoTIFF input files found")

    if skipped:
        print(f"Skipped {len(skipped)} unreadable/missing tile(s):")
        for skipped_tile in skipped:
            print(f" - {skipped_tile}")
    return _write_merged_tif(srcs, output_tif, compress=compress, quantize_step=quantize_step)