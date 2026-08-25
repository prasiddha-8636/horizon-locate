import os
import numpy as np
import pandas as pd
import rasterio
from osgeo import gdal
from rasterio.warp import reproject, Resampling
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interact
from pyprojroot import here

# Force GDAL to use Python exception handling instead of silent C++ crashes
gdal.UseExceptions()


def coordinate_to_pixel(geotransform, x, y, max_cols, max_rows):
    """
    Translates projected planar UTM Easting/Northing coordinates (X, Y) 
    into 2D raster array indices (Column, Row) with boundary clamping.
    
    Returns:
        col_clamped (int)
        row_clamped (int)
        is_inside (bool): True if the original coordinates fall inside the raster.
    """
    col = int((x - geotransform[0]) / geotransform[1])
    row = int((y - geotransform[3]) / geotransform[5])
    
    is_inside = (0 <= col < max_cols) and (0 <= row < max_rows)
    
    # Clamp indices to stay safely within the array limits to prevent fencepost errors
    col_clamped = max(0, min(col, max_cols - 1))
    row_clamped = max(0, min(row, max_rows - 1))
    
    return col_clamped, row_clamped, is_inside


def generate_viewshed_mask(src_ds, ox, oy, oz, max_distance_m):
    """
    Calculates a 3D line-of-sight viewshed on a raster using the GDAL 3.x C++ engine.
    """
    src_band = src_ds.GetRasterBand(1)
    temp_name = "temp_viewshed_run"
    
    dst_ds = gdal.ViewshedGenerate(
        srcBand=src_band,
        driverName="MEM",
        targetRasterName=temp_name,
        creationOptions=[],
        observerX=ox,
        observerY=oy,
        observerHeight=oz + 1.8,
        targetHeight=0.0,
        visibleVal=255,
        invisibleVal=0,
        outOfRangeVal=0,
        noDataVal=-9999,
        dfCurvCoeff=1.0,
        mode=1,
        maxDistance=max_distance_m
    )
    
    mask = dst_ds.GetRasterBand(1).ReadAsArray()
    
    # Force C++ GDAL engine to deallocate memory to prevent RAM accumulation
    dst_ds = None
    gdal.GetDriverByName('MEM').Delete(temp_name)
    
    return mask


def calculate_visible_distances(visibility_mask, observer_row, observer_col, resolution_m=90.0, max_distance_km=None):
    """
    Calculates Euclidean distances (in kilometers) from the observer 
    to all visible terrain pixels.
    """
    y, x = np.where(visibility_mask == 255)
    pixel_distances = np.sqrt((y - observer_row)**2 + (x - observer_col)**2)
    
    distances_km = (pixel_distances * resolution_m) / 1000.0
    
    if max_distance_km is not None:
        distances_km = distances_km[distances_km <= max_distance_km]
        
    return distances_km


def calculate_visibility_percentiles(distances, percentiles=(50, 75, 90, 95, 99), label="Visibility"):
    """
    Computes percentile visibility thresholds and prints a concise summary.

    Returns
    -------
    dict
        Mapping like {"p90": value_km, ...}
    """
    distances = np.asarray(distances, dtype=np.float64)
    if distances.size == 0:
        raise ValueError("distances is empty")

    values = np.percentile(distances, percentiles)
    stats = {f"p{int(p)}": float(v) for p, v in zip(percentiles, values)}

    return stats


def generate_lod_dem(src_ds, lod_path, resolution=360.0):
    """
    Pre-warps the high-resolution input raster into a lightweight Level-of-Detail (LOD)
    representation on disk using bilinear resampling to optimize interactive plotting speeds.
    """
    os.makedirs(os.path.dirname(lod_path), exist_ok=True)
    gdal.Warp(str(lod_path), src_ds, format='GTiff', xRes=resolution, yRes=resolution,
              resampleAlg='bilinear', creationOptions=["COMPRESS=DEFLATE"])


def crop_centered_window(array, center_row, center_col, radius_pixels, fill_value=0):
    """
    Crops a square window of size (2 * radius_pixels + 1) centered at (center_row, center_col).
    Pads with fill_value if the window extends outside the array boundaries.
    """
    size = 2 * radius_pixels + 1
    cropped = np.full((size, size), fill_value, dtype=array.dtype)
    
    src_row_start = max(0, center_row - radius_pixels)
    src_row_end = min(array.shape[0], center_row + radius_pixels + 1)
    src_col_start = max(0, center_col - radius_pixels)
    src_col_end = min(array.shape[1], center_col + radius_pixels + 1)
    
    dest_row_start = src_row_start - (center_row - radius_pixels)
    dest_row_end = dest_row_start + (src_row_end - src_row_start)
    dest_col_start = src_col_start - (center_col - radius_pixels)
    dest_col_end = dest_col_start + (src_col_end - src_col_start)
    
    cropped[dest_row_start:dest_row_end, dest_col_start:dest_col_end] = array[src_row_start:src_row_end, src_col_start:src_col_end]
    return cropped


def plot_radar_viewshed_panel(mask, oz, ox_km, oy_km, max_distance_km=500.0, rings=5):
    """
    Renders a concise, radar-style polar layout displaying visibility rings
    and cardinal directions centered on the active observer location.
    """
    extent_km = [-max_distance_km, max_distance_km, -max_distance_km, max_distance_km]
    fig, ax = plt.subplots(figsize=(10, 10), facecolor='#111111')
    ax.set_facecolor('#111111')
    
    ax.imshow(mask, cmap='gray', extent=extent_km, origin='upper')
    
    # Draw concentric radar distance rings
    for dist in np.arange(1, rings + 1) * (max_distance_km / rings):
        circle = plt.Circle((0, 0), dist, color='#39FF14', fill=False, linestyle='--', alpha=0.4, lw=1.2)
        ax.add_patch(circle)
        ax.text(10, dist + 2, f"{int(dist)}k", color='#39FF14', fontsize=9, alpha=0.7, fontweight='bold')
        
    ax.scatter(0, 0, color='red', edgecolors='white', s=100, zorder=5)
    
    # Concise compass directions
    ax.text(0, max_distance_km - 20, "N", color='white', ha='center', va='center', fontweight='bold', fontsize=16)
    ax.text(0, -max_distance_km + 20, "S", color='white', ha='center', va='center', fontweight='bold', fontsize=16)
    ax.text(max_distance_km - 20, 0, "E", color='white', ha='center', va='center', fontweight='bold', fontsize=16)
    ax.text(-max_distance_km + 20, 0, "W", color='white', ha='center', va='center', fontweight='bold', fontsize=16)
    
    ax.set_xlim(-max_distance_km, max_distance_km)
    ax.set_ylim(-max_distance_km, max_distance_km)
    ax.set_xlabel("Relative Easting (km)", color='white', fontsize=11)
    ax.set_ylabel("Relative Northing (km)", color='white', fontsize=11)
    ax.tick_params(colors='white')
    
    ax.set_title(f"Viewshed (Observer Elevation: {oz:.0f}m)", color='white', fontsize=12, fontweight='bold', pad=10)
    ax.grid(True, color='#39FF14', linestyle=':', alpha=0.15)
    plt.show()


def classify_altitude_band(z):
    """Classifies observer elevations into standard physiological/topographic Himalayan bands."""
    if z < 3000.0:
        return "< 3,000m"
    elif 3000.0 <= z < 4000.0:
        return "3,000m - 4,000m"
    elif 4000.0 <= z < 5000.0:
        return "4,000m - 5,000m"
    else:
        return "> 5,000m"


def render_viewshed_dashboard(dem_path, center_x, center_y, max_distance_km = 500.0):
    """Generates an ultra-fast, real-time viewshed dashboard using an in-memory LOD band."""
    src_ds = gdal.Open(dem_path)
    
    # Pre-compile the lightweight 360m LOD DEM on disk to bypass memory issues
    lod_path = here() / "notebooks" / "01_RegionStudy" / "output" / "dem_lod_360m.tif"
    generate_lod_dem(src_ds, lod_path, resolution=360.0)
    
    dx_slider = widgets.IntSlider(min=-50, max=50, step=5, value=0, description='dx (km):', continuous_update=False, layout=widgets.Layout(width='600px'))
    dy_slider = widgets.IntSlider(min=-50, max=50, step=5, value=0, description='dy (km):', continuous_update=False, layout=widgets.Layout(width='600px'))
    
    @interact(dx=dx_slider, dy=dy_slider)
    def plot_viewshed(dx=0, dy=0):
        ox = center_x + dx * 1000.0
        oy = center_y + dy * 1000.0
        
        lod_ds = gdal.Open(str(lod_path))
        lod_gt = lod_ds.GetGeoTransform()
        lod_array = lod_ds.GetRasterBand(1).ReadAsArray()
        
        col, row, is_inside = coordinate_to_pixel(lod_gt, ox, oy, lod_ds.RasterXSize, lod_ds.RasterYSize)
        oz = float(lod_array[row, col])
        
        mask = generate_viewshed_mask(lod_ds, ox, oy, oz, max_distance_m=max_distance_km * 1000.0)
        
        lod_ds = None  # Free file lock
        
        # Crop mask to center the observer and respect physical limits
        resolution_m = abs(lod_gt[1])
        radius_pixels = int((max_distance_km * 1000.0) / resolution_m)
        cropped_mask = crop_centered_window(mask, row, col, radius_pixels, fill_value=0)
        
        plot_radar_viewshed_panel(cropped_mask, oz, ox/1000.0, oy/1000.0, max_distance_km=max_distance_km)


def create_altitude_visibility_table(region, dem_path, grid_size=10, max_dist_km=500.0, return_total_horizons=False):
    """
    Evaluates viewsheds across a grid of observers and compiles a table 
    summarizing the visual horizon (maximum view range) per observer.
    """
    src_ds = gdal.Open(dem_path)
    gt = src_ds.GetGeoTransform()
    dem_array = src_ds.GetRasterBand(1).ReadAsArray()
    
    xs = np.linspace(region.west_m, region.east_m, grid_size)
    ys = np.linspace(region.south_m, region.north_m, grid_size)
    
    # Track the maximum horizon distance achieved by each observer
    bands = {
        "< 3,000m": [],
        "3,000m - 4,000m": [],
        "4,000m - 5,000m": [],
        "> 5,000m": [],
        "TOTAL (All Region Combined)": []
    }
    
    max_dist_m = max_dist_km * 1000.0

    for x in xs:
        for y in ys:
            col, row, is_inside = coordinate_to_pixel(gt, x, y, src_ds.RasterXSize, src_ds.RasterYSize)
            if not is_inside:
                continue
                
            z = float(dem_array[row, col])
            if z <= -9999 or np.isnan(z):
                continue
            
            mask = generate_viewshed_mask(src_ds, x, y, z, max_distance_m=max_dist_m)
            dist = calculate_visible_distances(mask, row, col, resolution_m=abs(gt[1]), max_distance_km=max_dist_km)
            
            if len(dist) == 0:
                continue
                
            # The "Horizon" for this specific observer is the furthest pixel they can see
            observer_horizon = float(np.nanmax(dist))
            
            band_name = classify_altitude_band(z)
            bands[band_name].append(observer_horizon)
            bands["TOTAL (All Region Combined)"].append(observer_horizon)
            
    # Compile the final statistics based on observer horizons
    table_rows = []
    for name, horizons in bands.items():
        if len(horizons) > 0:
            pct = calculate_visibility_percentiles(horizons, label=name)
            table_rows.append({
                "Observer Altitude": name,
                "Min Horizon (km)": np.min(horizons),
                "Max Horizon (km)": np.max(horizons),
                "Average Horizon (km)": np.mean(horizons),
                "P90 Horizon (km)": pct["p90"],
                "P95 Horizon (km)": pct["p95"],
                "P99 Horizon (km)": pct["p99"],
                "Sample Count": len(horizons)
            })
            
    df = pd.DataFrame(table_rows)
    df = df.round({
        "Min Horizon (km)": 1, 
        "Max Horizon (km)": 1, 
        "Average Horizon (km)": 1,
        "P90 Horizon (km)": 1,
        "P95 Horizon (km)": 1,
        "P99 Horizon (km)": 1,
    })

    if return_total_horizons:
        return df, np.asarray(bands["TOTAL (All Region Combined)"], dtype=np.float32)
    return df