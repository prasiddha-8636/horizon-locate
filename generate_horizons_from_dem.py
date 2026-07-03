#!/usr/bin/env python3
"""
Generate the Horizon Database from DEM using HORAYZON
======================================================

This script raycasts from 4,920 viewpoints (82×60 grid, 500m spacing) on the Khumbu DEM
and generates three horizon databases at different search radii:
  - Global:     80 km (coarse matching)
  - Local:      15 km (medium matching)
  - Restricted: 3 km (fine matching)

Output files:
  - data/horizon_db_global.npy (4920 × 1440)
  - data/horizon_db_local.npy (4920 × 1440)
  - data/horizon_db_restricted.npy (4920 × 1440)

Each row = horizon profile from one viewpoint
Each column = elevation angle at 0.25° azimuth resolution (0-359.75°)

RUNTIME: 2-6 hours on GPU (Colab T4), depends on accuracy settings
MEMORY: ~4 GB

USAGE:
  python generate_horizons_from_dem.py
"""

import os
import numpy as np
import rasterio
from pyproj import Transformer
import horayzon as hray
from tqdm import tqdm
import time

# ============================================================================
# CONFIGURATION
# ============================================================================

# Khumbu region bounds (matching unified_evaluation_pipeline.py)
KHUMBU_BOUNDS_GPS = {
    'min_lon': 86.582,
    'max_lon': 86.989,
    'min_lat': 27.770,
    'max_lat': 28.041
}

GRID_SPACING_M = 500.0  # Viewpoint spacing
UTM_ZONE = "EPSG:32645"  # UTM Zone 45N
GPS_CRS = "EPSG:4326"

# Raycast parameters
MAX_SEARCH_RADII = {
    "global": 80.0,      # km
    "local": 15.0,       # km
    "restricted": 3.0    # km
}

AZIMUTH_RESOLUTION = 0.25  # degrees
NUM_RAYS = int(360.0 / AZIMUTH_RESOLUTION)  # 1440 rays per viewpoint

CAMERA_HEIGHT_M = 1.8  # Virtual camera height above ground
VERTICAL_ACCURACY = 0.1  # Vertical accuracy in degrees

DEM_PATH = "data/dem.tif"


# ============================================================================
# SETUP & VALIDATION
# ============================================================================

def validate_inputs():
    """Check that DEM file exists and is readable."""
    if not os.path.exists(DEM_PATH):
        raise FileNotFoundError(f"DEM not found: {DEM_PATH}")
    print(f"✓ DEM file found: {DEM_PATH}")
    
    try:
        import horayzon
        print(f"✓ HORAYZON module loaded (version: {horayzon.__version__ if hasattr(horayzon, '__version__') else 'unknown'})")
    except ImportError:
        raise ImportError("HORAYZON not installed. Run: pip install horayzon")


def create_output_dir():
    """Create data directory if needed."""
    os.makedirs("data", exist_ok=True)


# ============================================================================
# BUILD VIEWPOINT GRID
# ============================================================================

def build_viewpoint_grid():
    """
    Build the current 4,920-viewpoint grid matching the existing horizon DB and mapping.
    
    Returns:
        viewpoints: (4920, 3) array [x_utm, y_utm, z_dem]
        Y_v, X_v: Grid coordinates
    """
    gps_to_utm = Transformer.from_crs(GPS_CRS, UTM_ZONE, always_xy=True)
    
    # Convert GPS bounds to UTM
    min_x, min_y = gps_to_utm.transform(KHUMBU_BOUNDS_GPS['min_lon'], KHUMBU_BOUNDS_GPS['min_lat'])
    max_x, max_y = gps_to_utm.transform(KHUMBU_BOUNDS_GPS['max_lon'], KHUMBU_BOUNDS_GPS['max_lat'])
    
    print(f"\nUTM bounds:")
    print(f"  X: {min_x:.1f} to {max_x:.1f} m")
    print(f"  Y: {min_y:.1f} to {max_y:.1f} m")
    
    # Build grid
    X_v = np.arange(min_x, max_x + GRID_SPACING_M, GRID_SPACING_M)
    Y_v = np.arange(min_y, max_y, GRID_SPACING_M)
    
    print(f"\nGrid dimensions:")
    print(f"  X: {len(X_v)} columns")
    print(f"  Y: {len(Y_v)} rows")
    print(f"  Total: {len(Y_v) * len(X_v)} viewpoints")
    
    # Create meshgrid
    view_xs, view_ys = np.meshgrid(X_v, Y_v)
    view_xs_ys = np.column_stack((view_xs.ravel(), view_ys.ravel())).astype(np.float32)
    
    return view_xs_ys, Y_v, X_v


def sample_dem_at_viewpoints(view_xs_ys):
    """
    Sample DEM elevations at viewpoint locations.
    
    Args:
        view_xs_ys: (N, 2) array of [utm_x, utm_y] coordinates
    
    Returns:
        viewpoints: (N, 3) array of [utm_x, utm_y, dem_z]
    """
    with rasterio.open(DEM_PATH) as src:
        dem_data = src.read(1).astype(np.float32)
        [pixel_width, _, start_x, _, pixel_height, start_y] = src.transform[:6]
        raw_ys = start_y + np.arange(src.height) * pixel_height
        
        # Handle inverted Y-axis
        if pixel_height < 0:
            raw_ys = raw_ys[::-1]
            pixel_height = -pixel_height
            dem_data = np.flipud(dem_data)

    # Convert UTM coordinates to DEM pixel indices
    view_cols = np.round((view_xs_ys[:, 0] - start_x) / pixel_width).astype(int)
    view_rows = np.round((view_xs_ys[:, 1] - raw_ys[0]) / pixel_height).astype(int)    
    
    # Clip to DEM bounds
    view_cols = np.clip(view_cols, 0, dem_data.shape[1] - 1)
    view_rows = np.clip(view_rows, 0, dem_data.shape[0] - 1)
    
    view_zs = dem_data[view_rows, view_cols].astype(np.float32)
    
    viewpoints = np.column_stack((view_xs_ys, view_zs)).astype(np.float32)
    
    print(f"\nViewpoint elevations:")
    print(f"  Min: {view_zs.min():.1f} m")
    print(f"  Max: {view_zs.max():.1f} m")
    print(f"  Mean: {view_zs.mean():.1f} m")
    
    return viewpoints


def build_terrain_mesh():
    """
    Build HORAYZON terrain mesh from DEM.
    
    Returns:
        vert_grid: Rearranged DEM buffer for HORAYZON
        height, width: DEM dimensions
    """
    with rasterio.open(DEM_PATH) as src:
        dem_data = src.read(1).astype(np.float32)
        [pixel_width, _, start_x, _, pixel_height, start_y] = src.transform[:6]
        
        xs = start_x + np.arange(src.width) * pixel_width
        ys = start_y + np.arange(src.height) * pixel_height
        
        if pixel_height < 0:
            ys = ys[::-1]
            pixel_height *= -1
            dem_data = np.flipud(dem_data)
        
        height, width = src.height, src.width
    
    # Build mesh for HORAYZON
    vert_grid = hray.auxiliary.rearrange_pad_buffer(
        *np.meshgrid(xs.astype(np.float32), ys.astype(np.float32)),
        dem_data
    )
    
    print(f"\nTerrain mesh built: {height} × {width} grid")
    
    return vert_grid, height, width, dem_data


# ============================================================================
# RAYCAST HORIZONS
# ============================================================================

def compute_horizons(vert_grid, height, width, viewpoints, max_search_radius_km):
    """
    Raycast horizon angles from all viewpoints using HORAYZON.
    
    Args:
        vert_grid: HORAYZON terrain mesh
        height, width: DEM dimensions
        viewpoints: (N, 3) array of [utm_x, utm_y, dem_z]
        max_search_radius_km: Search radius in km
    
    Returns:
        horizon_angles: (N, 1440) array of elevation angles in degrees
    """
    num_viewpoints = viewpoints.shape[0]
    
    # Reference vectors for HORAYZON
    vec_up = np.column_stack((
        np.zeros((num_viewpoints, 2)),
        np.ones(num_viewpoints)
    )).astype(np.float32)
    
    vec_north = np.column_stack((
        np.zeros(num_viewpoints),
        np.ones(num_viewpoints),
        np.zeros(num_viewpoints)
    )).astype(np.float32)
    
    ray_org_elev = np.full(num_viewpoints, CAMERA_HEIGHT_M, dtype=np.float32)
    
    print(f"\nRaycasting ({NUM_RAYS} rays × {num_viewpoints} viewpoints)...")
    print(f"  Search radius: {max_search_radius_km} km")
    print(f"  Vertical accuracy: {VERTICAL_ACCURACY}°")
    
    start_time = time.time()
    
    # Run HORAYZON horizon detection
    # Note: Newer versions of HORAYZON return 2 values instead of 3
    result = hray.horizon.horizon_locations(
        vert_grid,                    # 3D terrain model
        height,                       # DEM height
        width,                        # DEM width
        viewpoints,                   # Viewpoint locations
        vec_up,                       # Reference for elevation (zenith)
        vec_north,                    # Reference for azimuth (north)
        dist_search=max_search_radius_km,
        azim_num=NUM_RAYS,
        hori_acc=VERTICAL_ACCURACY,
        ray_org_elev=ray_org_elev,
        hori_dist_out=False,          # Don't compute distance (just angles)
        elev_ang_low_lim=-89
    )
    
    # Handle both old (3 return values) and new (2 return values) API
    if isinstance(result, tuple):
        if len(result) == 3:
            horizon_elevation_angles, _, _ = result
        elif len(result) == 2:
            horizon_elevation_angles, _ = result
        else:
            raise ValueError(f"Unexpected number of return values: {len(result)}")
    else:
        horizon_elevation_angles = result
    
    elapsed = time.time() - start_time
    print(f"  ✓ Raycast complete in {elapsed:.1f}s")
    
    # Convert radians to degrees
    horizon_angles = np.degrees(horizon_elevation_angles)
    
    print(f"  Output shape: {horizon_angles.shape}")
    print(f"  Value range: [{horizon_angles.min():.2f}°, {horizon_angles.max():.2f}°]")
    
    return horizon_angles


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("HORIZON DATABASE GENERATION (HORAYZON)")
    print("="*70)
    
    validate_inputs()
    create_output_dir()
    
    # Step 1: Build viewpoint grid
    print("\n[1/5] Building viewpoint grid...")
    view_xs_ys, Y_v, X_v = build_viewpoint_grid()
    
    # Step 2: Sample DEM at viewpoints
    print("\n[2/5] Sampling DEM at viewpoint locations...")
    viewpoints = sample_dem_at_viewpoints(view_xs_ys)
    
    # Step 3: Build terrain mesh
    print("\n[3/5] Building terrain mesh...")
    vert_grid, height, width, dem_data = build_terrain_mesh()
    
    # Step 4: Raycast horizons for each tier
    print("\n[4/5] Computing horizon profiles...")
    for tier_name, search_radius_km in MAX_SEARCH_RADII.items():
        print(f"\n  --- Tier: {tier_name.upper()} ({search_radius_km} km) ---")
        
        horizon_angles = compute_horizons(
            vert_grid, height, width, viewpoints, search_radius_km
        )
        
        # Save to disk
        output_path = f"data/horizon_db_{tier_name}.npy"
        np.save(output_path, horizon_angles)
        
        file_size_mb = os.path.getsize(output_path) / (1024**2)
        print(f"  ✓ Saved to {output_path} ({file_size_mb:.1f} MB)")
    
    # Step 5: Verify outputs
    print("\n[5/5] Verifying outputs...")
    for tier_name in MAX_SEARCH_RADII.keys():
        path = f"data/horizon_db_{tier_name}.npy"
        if os.path.exists(path):
            data = np.load(path)
            print(f"  ✓ {tier_name:12s}: {data.shape} → {data.dtype}")
        else:
            print(f"  ✗ {tier_name:12s}: MISSING")
    
    print("\n" + "="*70)
    print("✓ HORIZON DATABASE GENERATION COMPLETE")
    print("="*70)
    print("""
Next steps:
  1. Download all horizon_db_*.npy files to your local machine
  2. Place in data/ directory
  3. Run: python unified_evaluation_pipeline.py --mode diagnose
  4. Run: python unified_evaluation_pipeline.py --mode evaluate --limit 50
""")


if __name__ == "__main__":
    main()
