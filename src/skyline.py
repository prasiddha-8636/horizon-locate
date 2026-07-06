"""Generates the local horizon database from a regional Digital Elevation Model."""
import gc
from pathlib import Path
import numpy as np
import pandas as pd
from pyproj import Transformer, Geod
import rasterio
import rasterio.windows
import pyarrow as pa
import pyarrow.parquet as pq
import horayzon as hray
from tqdm import tqdm
from src.query_profile import is_profile_applicable


class SkylineDatabaseGenerator:
    def __init__(self, dem_file, region, dem_source="copernicus", ellps="WGS84",
                 dist_search_km=30.0, azim_num=360, hori_acc_deg=0.1,
                 eye_height_m=1.6, geoid="EGM96",
                 min_std_deg=1.5, min_max_elev_deg=1.0):
        self.dem_file = str(dem_file)
        self.dem_source = dem_source
        self.ellps = ellps
        self.dist_search_km = dist_search_km
        self.azim_num = azim_num
        self.hori_acc_deg = hori_acc_deg
        self.eye_height_m = eye_height_m
        self.geoid = geoid
        
        self.min_std_deg = min_std_deg
        self.min_max_elev_deg = min_max_elev_deg
        self.region = region

    def _meters_to_degrees(self, spacing_m, mean_lon_deg, mean_lat_deg):
        geod = Geod(ellps=self.ellps)
        lon_end, _, _ = geod.fwd(mean_lon_deg, mean_lat_deg, 90.0, spacing_m)
        dlon_deg = abs(lon_end - mean_lon_deg)
        _, lat_end, _ = geod.fwd(mean_lon_deg, mean_lat_deg, 0.0, spacing_m)
        dlat_deg = abs(lat_end - mean_lat_deg)
        return dlon_deg, dlat_deg

    def generate_database(self, output_path, grid_spacing_m=30,
                          mesh_output_path=None, meta_output_path=None,
                          start_idx=0, end_idx=None, batch_size=4096,
                          save_raw_horizon=False, raw_horizon_stride=1,
                          raw_horizon_decimals=2):
        """Renders, filters, and saves horizon profiles across the target grid."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        mean_lon = (self.region.west_deg + self.region.east_deg) / 2.0
        mean_lat = (self.region.south_deg + self.region.north_deg) / 2.0
        dlon, dlat = self._meters_to_degrees(grid_spacing_m, mean_lon, mean_lat)
        n_lon = int((self.region.east_deg - self.region.west_deg) / dlon)
        n_lat = int((self.region.north_deg - self.region.south_deg) / dlat)

        print(f"Loading DEM geometry from: {self.dem_file}")
        with rasterio.open(self.dem_file) as src:
            dem_crs = src.crs
            dlon_buf, dlat_buf = self._meters_to_degrees(self.dist_search_km * 1000.0, mean_lon, mean_lat)
            
            west_buf = self.region.west_deg - dlon_buf
            east_buf = self.region.east_deg + dlon_buf
            south_buf = self.region.south_deg - dlat_buf
            north_buf = self.region.north_deg + dlat_buf
            
            to_dem_crs = Transformer.from_crs("EPSG:4326", dem_crs, always_xy=True)
            xs_buf, ys_buf = to_dem_crs.transform(
                [west_buf, east_buf, west_buf, east_buf],
                [south_buf, south_buf, north_buf, north_buf]
            )
            min_x_dem, max_x_dem = min(xs_buf), max(xs_buf)
            min_y_dem, max_y_dem = min(ys_buf), max(ys_buf)
            
            raw_window = rasterio.windows.from_bounds(
                min_x_dem, min_y_dem, max_x_dem, max_y_dem, transform=src.transform
            )
            
            row_start = int(max(0, np.floor(raw_window.row_off)))
            row_end = int(min(src.height, np.ceil(raw_window.row_off + raw_window.height)))
            col_start = int(max(0, np.floor(raw_window.col_off)))
            col_end = int(min(src.width, np.ceil(raw_window.col_off + raw_window.width)))
            
            window = rasterio.windows.Window(col_start, row_start, col_end - col_start, row_end - row_start)
            dem_data = src.read(1, window=window).astype(np.float32)
            
            if src.nodata is not None and np.isfinite(src.nodata):
                dem_data[dem_data == src.nodata] = np.nan
            dem_data = np.nan_to_num(dem_data, nan=0.0, posinf=0.0, neginf=0.0)
            valid_mask = dem_data > 10.0
            if np.any(valid_mask):
                min_valid = float(np.min(dem_data[valid_mask]))
                dem_data[~valid_mask] = min_valid

            window_transform = src.window_transform(window)
            px_w, _, start_x, _, px_h, start_y = window_transform[:6]
            xs = start_x + np.arange(window.width) * px_w
            ys = start_y + np.arange(window.height) * px_h

            x_mesh_dem, y_mesh_dem = np.meshgrid(xs.astype(np.float32), ys.astype(np.float32))
            to_utm = Transformer.from_crs(dem_crs, self.region.epsg, always_xy=True)
            x_mesh, y_mesh = to_utm.transform(x_mesh_dem, y_mesh_dem)

            vert_grid = hray.auxiliary.rearrange_pad_buffer(
                x_mesh.astype(np.float32), 
                y_mesh.astype(np.float32), 
                dem_data
            )
            dem_dim_0, dem_dim_1 = dem_data.shape

        if mesh_output_path:
            mesh_output_path = Path(mesh_output_path)
            mesh_output_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(mesh_output_path, vert_grid)

        if meta_output_path:
            meta_output_path = Path(meta_output_path)
            meta_output_path.parent.mkdir(parents=True, exist_ok=True)
            meta_dict = {
                "dem_dim_0": dem_dim_0,
                "dem_dim_1": dem_dim_1,
                "px_w": float(px_w),
                "px_h": float(px_h),
                "start_x": float(xs[0]),
                "start_y": float(ys[0])
            }
            with open(meta_output_path, "w") as f:
                json.dump(meta_dict, f, indent=4)

        lon_grid = np.arange(self.region.west_deg, self.region.east_deg, dlon)
        lat_grid = np.arange(self.region.south_deg, self.region.north_deg, dlat)
        lon_loc, lat_loc = np.meshgrid(lon_grid, lat_grid)
        lon_loc = lon_loc.ravel()
        lat_loc = lat_loc.ravel()

        total_points = len(lon_loc)
        start_idx = int(max(0, start_idx))
        end_idx = total_points if end_idx is None else int(min(total_points, end_idx))
        target_count = end_idx - start_idx
        
        if target_count <= 0:
            return str(output_path)

        to_utm_crs = Transformer.from_crs("EPSG:4326", self.region.epsg, always_xy=True)
        xs_proj, ys_proj = to_utm_crs.transform(lon_loc[start_idx:end_idx], lat_loc[start_idx:end_idx])
        
        to_dem_crs = Transformer.from_crs("EPSG:4326", dem_crs, always_xy=True)
        xs_dem_coords, ys_dem_coords = to_dem_crs.transform(lon_loc[start_idx:end_idx], lat_loc[start_idx:end_idx])

        cols = np.round((xs_dem_coords - xs[0]) / px_w).astype(int)
        rows = np.round((ys_dem_coords - ys[0]) / px_h).astype(int)
        cols = np.clip(cols, 0, dem_dim_1 - 1)
        rows = np.clip(rows, 0, dem_dim_0 - 1)
        zs_proj = dem_data[rows, cols]

        writer = None
        skipped_flat_count = 0
        written_count = 0
        
        try:
            # Wrap raycasting loop in tqdm
            for i in tqdm(range(0, target_count, batch_size), desc="Ray-casting Skyline Grid"):
                chunk_end = min(i + batch_size, target_count)
                chunk_size = chunk_end - i

                coords = np.column_stack((
                    xs_proj[i:chunk_end],
                    ys_proj[i:chunk_end],
                    zs_proj[i:chunk_end]
                )).astype(np.float32)

                vec_up = np.zeros((chunk_size, 3), dtype=np.float32)
                vec_up[:, 2] = 1.0
                vec_north = np.zeros((chunk_size, 3), dtype=np.float32)
                vec_north[:, 1] = 1.0

                ray_org_elev = np.full(chunk_size, self.eye_height_m, dtype=np.float32)

                result = hray.horizon.horizon_locations(
                    vert_grid, dem_dim_0, dem_dim_1,
                    coords, vec_up, vec_north,
                    dist_search=self.dist_search_km * 1000.0,
                    azim_num=self.azim_num,
                    hori_acc=self.hori_acc_deg,
                    ray_org_elev=ray_org_elev,
                    hori_dist_out=False,
                    elev_ang_low_lim=-89.0
                )
                if isinstance(result, tuple):
                    horizon_elev_rad = result[0]
                    azim_rad = result[2] if len(result) == 3 else None
                else:
                    horizon_elev_rad = result
                    azim_rad = None

                theta_deg = np.rad2deg(horizon_elev_rad)
                azim_deg = np.rad2deg(azim_rad) if azim_rad is not None else None

                batch_records = []
                for idx in range(chunk_size):
                    is_valid, _ = is_profile_applicable(
                        theta_deg[idx], 
                        min_std_deg=self.min_std_deg, 
                        min_max_elev_deg=self.min_max_elev_deg
                    )
                    
                    if not is_valid:
                        skipped_flat_count += 1
                        continue

                    rec = {
                        "lon": float(lon_loc[start_idx + i + idx]),
                        "lat": float(lat_loc[start_idx + i + idx]),
                        "elevation_m": float(zs_proj[i + idx])
                    }
                    
                    if save_raw_horizon:
                        raw_stride = max(1, int(raw_horizon_stride))
                        raw_curve = theta_deg[idx, ::raw_stride]
                        raw_curve = np.round(raw_curve, int(raw_horizon_decimals))
                        rec["raw_horizon_deg"] = raw_curve.astype(np.float32).tolist()
                        if azim_deg is not None:
                            rec["raw_azimuth_deg"] = np.round(
                                azim_deg[::raw_stride],
                                int(raw_horizon_decimals),
                            ).astype(np.float32).tolist()
                            
                    batch_records.append(rec)

                if batch_records:
                    batch_table = pa.Table.from_pandas(pd.DataFrame(batch_records), preserve_index=False)
                    if writer is None:
                        writer = pq.ParquetWriter(str(output_path), batch_table.schema)
                    writer.write_table(batch_table)
                    written_count += len(batch_records)
                    del batch_records, batch_table
                
                gc.collect()

            print(f"✓ Saved {written_count:,} viewpoints (skipped {skipped_flat_count:,} flat/unshielded sites) to {output_path}")
        finally:
            if writer is not None:
                writer.close()

        return str(output_path)