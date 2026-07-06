"""Extracts and validates 1D elevation-angle skyline profiles from terrain masks."""
import numpy as np
import cv2
from PIL import Image
import scipy.ndimage as ndimage


def is_profile_applicable(profile, min_std_deg=1.5, min_max_elev_deg=1.0):
    """
    Evaluates whether a horizon profile contains sufficient topographic variation 
    and vertical relief to be reliably matched.
    """
    profile = np.asarray(profile, dtype=np.float64)
    if profile.size == 0:
        return False, "Empty profile"
        
    std_val = np.std(profile)
    max_val = np.max(profile)
    
    if std_val < min_std_deg:
        return False, f"Profile too flat (std={std_val:.2f}° < {min_std_deg}°)"
        
    if max_val < min_max_elev_deg:
        return False, f"Insufficient terrain relief above horizontal (max={max_val:.2f}° < {min_max_elev_deg}°)"
        
    return True, "Valid topographic profile"


def extract_elevation_profile(mask_path, fov_y_deg=65.0, aspect_ratio=None, r_tilt=None, bin_deg=0.25):
    """
    Translates a binary sky-terrain mask into a 1D elevation-angle profile 
    projected onto a uniform azimuth grid.
    """
    mask = np.array(Image.open(mask_path).convert("L"))
    H, W = mask.shape
    aspect_ratio = aspect_ratio or (W / H)

    sky_is_white = np.mean(mask[:10, :]) > np.mean(mask[-10:, :])
    binary = (mask < 128).astype(np.uint8) if sky_is_white else (mask >= 128).astype(np.uint8)

    skyline_px = np.array([
        (np.where(binary[:, c] == 1)[0][0] if np.any(binary[:, c] == 1) else H - 1)
        for c in range(W)
    ], dtype=np.float32)

    skyline_px = ndimage.median_filter(skyline_px, size=5)

    hfov_deg = np.degrees(2.0 * np.arctan(np.tan(np.radians(fov_y_deg) / 2.0) * aspect_ratio))
    focal_x = W / (2.0 * np.tan(np.radians(hfov_deg) / 2.0))
    focal_y = H / (2.0 * np.tan(np.radians(fov_y_deg) / 2.0))

    x_c, y_c = W / 2.0, H / 2.0
    cols = np.arange(W)
    rays = np.vstack([(cols - x_c) / focal_x, (y_c - skyline_px) / focal_y, -np.ones(W)])
    rays /= np.linalg.norm(rays, axis=0)

    if r_tilt is not None:
        rays = np.asarray(r_tilt) @ rays
        
    elev_deg = np.degrees(np.arcsin(np.clip(rays[1, :], -1.0, 1.0)))
    azim_deg = np.degrees(np.arctan2(rays[0, :], -rays[2, :]))

    order = np.argsort(azim_deg)
    azim_deg, elev_deg = azim_deg[order], elev_deg[order]

    start_az = np.ceil(azim_deg[0] / bin_deg) * bin_deg
    end_az = np.floor(azim_deg[-1] / bin_deg) * bin_deg
    grid = np.arange(start_az, end_az + 1e-6, bin_deg)
    profile = np.interp(grid, azim_deg, elev_deg)
    
    return profile, float(start_az)