"""Memory-efficient horizon visualization explorer with aligned camera optics and pitch."""
from pathlib import Path
import os
import gc
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
import pyarrow.parquet as pq
import ipywidgets as widgets
from IPython.display import display, clear_output
from PIL import Image

from src.mountain_engine import MountainEngine


def skyline_dataset_paths(root, mode="coarse"):
    root = Path(root)
    
    return {
        "dem_file": root / "data" / "digital_elevation_model" / "dem_30m.tif",
        "db_file": root / "notebooks" / "02_SkylineDatabase" / "output" / "skyline_db.parquet",
        "mesh_file": root / "notebooks" / "02_SkylineDatabase" / "output" / "terrain_mesh.npy",
        "meta_file": root / "notebooks" / "02_SkylineDatabase" / "output" / "terrain_meta.json",
    }


class SkylineHorizonExplorer:
    def __init__(self, dem_file, db_file, mesh_file=None, meta_file=None):
        self.dem_file = Path(dem_file)
        self.db_file = Path(db_file)
        self.mesh_file = None if mesh_file is None else Path(mesh_file)
        self.meta_file = None if meta_file is None else Path(meta_file)

        # 1. Open Parquet reader and load ONLY the lightweight coordinate columns (takes ~2 MB RAM)
        self.parquet_file = pq.ParquetFile(self.db_file)
        schema = self.parquet_file.schema.to_arrow_schema()
        self.columns_in_file = schema.names
        
        # Determine coordinate elevation column name
        elev_col = "elevation_m" if "elevation_m" in self.columns_in_file else "elevation"
        
        # Read only light metadata columns
        self.df = self.parquet_file.read(columns=["lon", "lat", elev_col]).to_pandas().reset_index(drop=True)
        self.df["coeff_1"] = self.df[elev_col]  # Fallback visualization coefficient

        with rasterio.open(self.dem_file) as src:
            self.dem_crs = src.crs
            self.dem_bounds = src.bounds

        to_dem = Transformer.from_crs("EPSG:4326", self.dem_crs, always_xy=True)
        x_dem, y_dem = to_dem.transform(self.df["lon"].values, self.df["lat"].values)
        self.df["x_dem"] = x_dem
        self.df["y_dem"] = y_dem

        self.grid = self.df.pivot_table(index="lat", columns="lon", values="coeff_1")

    def print_summary(self):
        print(f"DEM: {self.dem_file}")
        print(f"DB: {self.db_file}")
        print(f"Viewpoints: {len(self.df):,}")
        print(f"Grid size: {self.grid.shape[0]} x {self.grid.shape[1]}")

        if self.mesh_file and self.mesh_file.exists():
            mesh_arr = np.load(self.mesh_file, mmap_mode="r")
            print(f"Mesh buffer shape (memory-mapped): {mesh_arr.shape}")

        if self.meta_file and self.meta_file.exists():
            with open(self.meta_file, "r") as f:
                meta = json.load(f)
            print(f"Meta dims: ({meta.get('dem_dim_0')}, {meta.get('dem_dim_1')})")

    def plot_sample(
        self,
        sample_idx=0,
        cam_azimuth_deg=35.0,
        cam_pitch_deg=0.0,
        cam_fov_deg=60.0,
        look_distance_km=5.0,
        eye_height_m=1.6,
        mesh_stride=2,
        render_w=480,
        render_h=320,
        render_3d=True,
    ):
        # Read pre-projected coordinates from in-memory metadata dataframe
        tx = float(self.df.loc[int(sample_idx), "x_dem"])
        ty = float(self.df.loc[int(sample_idx), "y_dem"])
        lon_val = float(self.df.loc[int(sample_idx), "lon"])
        lat_val = float(self.df.loc[int(sample_idx), "lat"])

        # Find which Parquet Row Group contains our target sample_idx viewpoint
        cumulative_rows = 0
        target_row_group = 0
        row_offset_in_group = 0
        
        for rg_idx in range(self.parquet_file.num_row_groups):
            num_rows_in_rg = self.parquet_file.metadata.row_group(rg_idx).num_rows
            if cumulative_rows <= int(sample_idx) < cumulative_rows + num_rows_in_rg:
                target_row_group = rg_idx
                row_offset_in_group = int(sample_idx) - cumulative_rows
                break
            cumulative_rows += num_rows_in_rg

        # Lazily load only that single Row Group from disk
        single_rg_table = self.parquet_file.read_row_group(target_row_group, columns=["raw_horizon_deg", "raw_azimuth_deg"])
        row = single_rg_table.to_pandas().iloc[row_offset_in_group]

        img = None
        if render_3d:
            try:
                # Calculate correct Vertical FOV (yfov) from the Horizontal FOV slider value
                aspect_ratio = float(render_w) / float(render_h)
                hfov_rad = np.radians(cam_fov_deg)
                yfov_rad = 2.0 * np.arctan(np.tan(hfov_rad / 2.0) / aspect_ratio)

                engine = MountainEngine(
                    dem_path=str(self.dem_file),
                    texture_path=None,
                    dem_stride=int(mesh_stride),
                    render_width=int(render_w),
                    render_height=int(render_h),
                    crop_center_xy=(tx, ty),
                    crop_radius_m=float(look_distance_km) * 1000.0 / np.cos(np.deg2rad(cam_fov_deg / 2)) * 2.0
                )
                img = engine.get_render_first_person(
                    eye_xy=(tx, ty),
                    azimuth_deg=float(cam_azimuth_deg),
                    pitch_deg=float(cam_pitch_deg),  # Pass vertical camera tilt
                    eye_height_m=float(eye_height_m),
                    yfov=yfov_rad,
                    look_distance_m=float(look_distance_km) * 1000.0,
                )
                engine.close()
                del engine
                gc.collect()
            except Exception as e:
                print(f"[3D render skipped: {e}]")
                img = None

        has_raw_horizon = "raw_horizon_deg" in self.columns_in_file
        raw_curve = np.array(row["raw_horizon_deg"], dtype=np.float32) if has_raw_horizon else None
        
        has_raw_azimuth = "raw_azimuth_deg" in self.columns_in_file
        if has_raw_azimuth:
            azim_deg = np.array(row["raw_azimuth_deg"], dtype=np.float32)
        elif has_raw_horizon:
            azim_deg = np.linspace(0.0, 360.0, len(raw_curve), endpoint=False, dtype=np.float32)
        else:
            azim_deg = None

        n_panels = 3 if img is not None else 2
        fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5.5))
        if n_panels == 2:
            axes = [None, axes[0], axes[1]]

        if img is not None:
            axes[0].imshow(img)
            axes[0].set_title(f"First-person render (az={cam_azimuth_deg:.0f}°, hfov={cam_fov_deg:.0f}°)")
            axes[0].axis("off")

        if has_raw_horizon:
            axes[1].plot(azim_deg, raw_curve, lw=2, color="tab:blue")
            axes[1].set_title("Actual Horizon Curve (Elevation Angle)")
            axes[1].set_xlabel("Azimuth (deg, clockwise from North)")
            axes[1].set_ylabel("Elevation Angle (deg)")
            axes[1].set_xlim(0.0, 360.0)

            if img is not None:
                self._draw_camera_fov(axes[1], cam_azimuth_deg, cam_pitch_deg, cam_fov_deg, render_w, render_h, azim_deg, raw_curve)
        else:
            axes[1].text(0.5, 0.5, "No Horizon Curve Stored", ha='center', va='center')
            axes[1].set_title("Horizon Curve Missing")
            
        axes[1].grid(alpha=0.3)

        im = axes[2].imshow(
            self.grid.values,
            origin="lower",
            aspect="auto",
            extent=[
                self.grid.columns.min(),
                self.grid.columns.max(),
                self.grid.index.min(),
                self.grid.index.max(),
            ],
            cmap="viridis",
        )
        axes[2].scatter([lon_val], [lat_val], c="red", s=40, label="selected")
        axes[2].set_title("Generated Horizon Grid (Elevation)")
        axes[2].set_xlabel("Longitude")
        axes[2].set_ylabel("Latitude")
        axes[2].legend(loc="upper right")
        fig.colorbar(im, ax=axes[2], shrink=0.85, label="Elevation (m)")

        plt.tight_layout()
        plt.show()
        plt.close(fig)

    @staticmethod
    def _draw_camera_fov(ax, center_az, center_pitch, fov_deg, render_w, render_h, azim_deg, raw_curve):
        """Draws the camera viewport box using robust data coordinates centered on camera pitch."""
        lo = center_az - fov_deg / 2.0
        hi = center_az + fov_deg / 2.0

        # Calculate exact vertical FOV from the horizontal FOV based on render aspect ratio
        aspect_ratio = float(render_w) / float(render_h)
        vfov_deg = np.degrees(2.0 * np.arctan(np.tan(np.radians(fov_deg) / 2.0) / aspect_ratio))
        
        # Center the box vertically at the exact physical pitch angle of the camera
        elo, ehi = center_pitch - vfov_deg / 2.0, center_pitch + vfov_deg / 2.0

        def plot_viewport_rect(l, r):
            # Draw boundary lines in pure data coordinates
            ax.plot([l, r], [elo, elo], color="darkorange", lw=1.8, zorder=3)
            ax.plot([l, r], [ehi, ehi], color="darkorange", lw=1.8, zorder=3)
            ax.plot([l, l], [elo, ehi], color="darkorange", lw=1.8, zorder=3)
            ax.plot([r, r], [elo, ehi], color="darkorange", lw=1.8, zorder=3)
            
            # Fill the viewport interior
            ax.fill_between([l, r], elo, ehi, color="orange", alpha=0.12, zorder=1)

        # Handle circular wrap-around seamlessly
        if lo < 0:
            plot_viewport_rect(0, hi)
            plot_viewport_rect(360 + lo, 360)
        elif hi > 360:
            plot_viewport_rect(lo, 360)
            plot_viewport_rect(0, hi - 360)
        else:
            plot_viewport_rect(lo, hi)

        # Draw the optical center lines
        ax.axvline(center_az, color="darkorange", lw=1, ls="--", alpha=0.8)
        ax.axhline(center_pitch, color="darkorange", lw=1, ls="--", alpha=0.5)

    def interactive_widget(self):
        # Define clean, standard sliders
        sliders = {
            "sample_idx": widgets.IntSlider(min=0, max=len(self.df) - 1, step=1, value=0, description="Index"),
            "cam_azimuth_deg": widgets.FloatSlider(min=0.0, max=359.0, step=5.0, value=35.0, description="Heading (°)"),
            "cam_pitch_deg": widgets.FloatSlider(min=-45.0, max=45.0, step=2.0, value=0.0, description="Pitch (°)"),
            "cam_fov_deg": widgets.FloatSlider(min=20.0, max=120.0, step=5.0, value=60.0, description="FOV (°)"),
            "look_distance_km": widgets.FloatSlider(min=1.0, max=15.0, step=1.0, value=5.0, description="Distance (km)"),
            "eye_height_m": widgets.FloatSlider(min=1.0, max=3.0, step=0.1, value=1.6, description="Eye Height (m)"),
            "mesh_stride": widgets.IntSlider(min=1, max=10, step=1, value=2, description="Mesh Stride"),
            "render_3d": widgets.Checkbox(value=True, description="Render 3D Mesh Preview"),
        }

        # Format layout boxes to group controls logically into columns
        col_position = widgets.VBox([
            widgets.HTML("<h4><b>[ 1. Viewpoint ]</b></h4>"),
            sliders["sample_idx"],
            sliders["eye_height_m"]
        ], layout=widgets.Layout(padding='10px', border='1px solid #ccc', margin='5px', width='32%'))

        col_camera = widgets.VBox([
            widgets.HTML("<h4><b>[ 2. Camera Navigation ]</b></h4>"),
            sliders["cam_azimuth_deg"],
            sliders["cam_pitch_deg"],
            sliders["cam_fov_deg"]
        ], layout=widgets.Layout(padding='10px', border='1px solid #ccc', margin='5px', width='36%'))

        col_render = widgets.VBox([
            widgets.HTML("<h4><b>[ 3. Quality Settings ]</b></h4>"),
            sliders["look_distance_km"],
            sliders["mesh_stride"],
            sliders["render_3d"]
        ], layout=widgets.Layout(padding='10px', border='1px solid #ccc', margin='5px', width='32%'))

        control_panel = widgets.HBox([col_position, col_camera, col_render], layout=widgets.Layout(width='100%'))

        out = widgets.Output()

        def redraw(*_):
            kwargs = {name: w.value for name, w in sliders.items()}
            with out:
                clear_output(wait=True)
                self.plot_sample(**kwargs)

        for w in sliders.values():
            w.observe(redraw, names="value")

        redraw()
        return widgets.VBox([control_panel, out])


def plot_synthetic_samples(images_dir, masks_dir):
    """
    Renders an interactive side-by-side plot panel in the notebook 
    to visually traverse and verify generated query images and sky masks.
    """
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)
    
    # Count generated files dynamically
    sample_files = sorted([
        f for f in os.listdir(images_dir) 
        if f.lower().endswith(".png") and f.startswith("sample_")
    ])
    max_samples = len(sample_files)

    if max_samples == 0:
        print(f"[Warning] No generated sample PNG files found in: {images_dir}")
        return

    # Define simple, clean selection slider
    sample_slider = widgets.IntSlider(
        min=0, 
        max=max_samples - 1, 
        step=1, 
        value=0, 
        description="Sample ID",
        layout=widgets.Layout(width="50%")
    )

    out = widgets.Output()

    def display_sample(change):
        sample_id = change if isinstance(change, int) else change["new"]
        
        img_path = images_dir / f"sample_{sample_id:04d}.png"
        mask_path = masks_dir / f"sample_{sample_id:04d}.png"
        
        if not img_path.exists() or not mask_path.exists():
            with out:
                clear_output(wait=True)
                print(f"Sample {sample_id} files not found on disk.")
            return

        img = Image.open(img_path)
        mask = Image.open(mask_path)
        
        with out:
            clear_output(wait=True)
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            
            axes[0].imshow(img)
            axes[0].set_title(f"Generated Query View (ID: {sample_id})", fontsize=11, fontweight='bold')
            axes[0].axis('off')
            
            axes[1].imshow(mask, cmap='gray')
            axes[1].set_title("Generated Sky Mask\n(Terrain=255/White, Sky=0/Black)", fontsize=11, fontweight='bold')
            axes[1].axis('off')
            
            plt.tight_layout()
            plt.show()
            plt.close(fig)

    sample_slider.observe(display_sample, names="value")

    # Initial render of the first sample
    display_sample(0)

    # Render dashboard components inside the notebook output area
    display(widgets.VBox([sample_slider, out]))