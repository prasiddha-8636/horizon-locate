from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class PipelineConfig:
    # Region
    region_json: str = "notebooks/01_RegionStudy/output/actual_bounds.json"

    # DEM
    dem_source: str = "copernicus"
    dem_output: str = "data/digital_elevation_model/dem_30m.tif"

    # Skyline DB
    dist_search_km: float = 30.0
    azim_num: int = 720
    hori_acc_deg: float = 0.1
    eye_height_m: float = 1.6
    grid_spacing_m: float = 30.0
    batch_size: int = 4096
    min_std_deg: float = 1.5
    min_max_elev_deg: float = 1.0

    # Segmentation
    seg_model_path: str = "data/sky_segmentation_unet_model.pth"
    seg_input_size: int = 256
    seg_device: Optional[str] = None
    min_sky_ratio: float = 0.05
    max_sky_ratio: float = 0.95
    min_boundary_coverage: float = 0.5

    # Profile
    fov_y_deg: float = 65.0
    bin_deg: float = 0.5
    median_kernel: int = 5

    # Matching
    fft_weights: Tuple[float, float] = (0.5, 0.5)
    min_corr: float = 0.30
    min_score_gap: float = 0.03
    top_k: int = 5
    dtw_window: int = 15
    spatial_stride: int = 12

    # Evaluation
    chunk_rows: int = 4000
    correct_dist_m: float = 500.0
    height_tolerance_m: float = 200.0
    compass_tolerance_deg: float = 20.0
