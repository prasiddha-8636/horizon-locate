"""Topographic and optical physics models for horizon visibility validation."""
import numpy as np
import matplotlib.pyplot as plt


def calculate_curvature_drop(distance_km, k_refraction=0.13):
    """Calculates the terrain height hidden by Earth's curvature."""
    R_earth = 6371000.0  # Earth's radius in meters
    distance_m = np.asarray(distance_km, dtype=np.float64) * 1000.0
    return (distance_m ** 2) * (1.0 - k_refraction) / (2.0 * R_earth)


def calculate_atmospheric_contrast(distance_km, visibility_km=50.0):
    """Calculates image clarity loss over distance."""
    extinction_coeff = 3.912 / float(visibility_km)
    return np.exp(-extinction_coeff * np.asarray(distance_km, dtype=np.float64))


def plot_horizon_limit_proof(distances_km, visibility_km=50.0):
    """Plots Earth curvature drop and visibility loss side-by-side."""
    distances_km = np.asarray(distances_km, dtype=np.float64)
    
    curvature_drop_m = calculate_curvature_drop(distances_km)
    clarity_percent = calculate_atmospheric_contrast(distances_km, visibility_km) * 100.0

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Earth Curvature Drop
    axes[0].plot(distances_km, curvature_drop_m, color='crimson', lw=2.5)
    axes[0].axvline(30, color='gray', linestyle='--')
    axes[0].text(33, np.max(curvature_drop_m) * 0.1, "30 km limit", color='gray', fontsize=10)
    axes[0].set_title("Elevation Hidden due to Earth's Curvature", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Distance (km)", fontsize=10)
    axes[0].set_ylabel("Hidden Height (meters)", fontsize=10)
    axes[0].grid(True, linestyle=':', alpha=0.5)

    # Plot 2: Atmospheric Visibility
    axes[1].plot(distances_km, clarity_percent, color='teal', lw=2.5)
    axes[1].axvline(30, color='gray', linestyle='--')
    axes[1].text(33, 80, "30 km limit", color='gray', fontsize=10)
    axes[1].set_title("Atmospheric Visibility", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Distance (km)", fontsize=10)
    axes[1].set_ylabel("Image Clarity (%)", fontsize=10)
    axes[1].grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.show()