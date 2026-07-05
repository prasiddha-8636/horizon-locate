"""3D perspective rendering engine utilizing smooth bilinear terrain sampling."""
import os
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import rasterio
import trimesh
import pyrender
from pyproj import Transformer
from PIL import Image
import rasterio.windows


class MountainEngine:
    def __init__(
        self,
        dem_path,
        texture_path=None,
        dem_stride=5,
        render_width=800,
        render_height=600,
        camera_height_m=500.0,
        crop_center_xy=None,
        crop_radius_m=20000.0,
    ):
        self.dem_path = dem_path
        self.texture_path = texture_path
        self.dem_stride = max(1, int(dem_stride))
        self.camera_height_m = float(camera_height_m)

        with rasterio.open(dem_path) as src:
            self.dem_crs = src.crs

            if crop_center_xy is not None:
                cx, cy = crop_center_xy
                window = rasterio.windows.from_bounds(
                    cx - crop_radius_m, cy - crop_radius_m,
                    cx + crop_radius_m, cy + crop_radius_m,
                    transform=src.transform,
                )
                row_start = int(max(0, np.floor(window.row_off)))
                row_end = int(min(src.height, np.ceil(window.row_off + window.height)))
                col_start = int(max(0, np.floor(window.col_off)))
                col_end = int(min(src.width, np.ceil(window.col_off + window.width)))
                window = rasterio.windows.Window(col_start, row_start, col_end - col_start, row_end - row_start)
                win_transform = src.window_transform(window)

                out_h = max(1, int(window.height) // self.dem_stride)
                out_w = max(1, int(window.width) // self.dem_stride)
                raw_dem = src.read(1, window=window, out_shape=(out_h, out_w)).astype(np.float32)

                px_w = win_transform.a
                px_h = win_transform.e
                start_x = win_transform.c
                start_y = win_transform.f
                eff_px_w = px_w * (window.width / out_w)
                eff_px_h = px_h * (window.height / out_h)
                xs_raw = start_x + (np.arange(out_w) + 0.5) * eff_px_w
                ys_raw = start_y + (np.arange(out_h) + 0.5) * eff_px_h
                b_left, b_right = start_x, start_x + window.width * px_w
                b_bottom, b_top = start_y + window.height * px_h, start_y
                if px_h < 0:
                    b_bottom, b_top = b_top, b_bottom
            else:
                raw_dem = src.read(1)[:: self.dem_stride, :: self.dem_stride].astype(np.float32)
                b = src.bounds
                xs_raw = np.linspace(b.left, b.right, raw_dem.shape[1])
                ys_raw = np.linspace(b.bottom, b.top, raw_dem.shape[0])
                b_left, b_right, b_bottom, b_top = b.left, b.right, b.bottom, b.top

            if ys_raw[0] > ys_raw[-1]:
                ys_raw = ys_raw[::-1]
                raw_dem = np.flipud(raw_dem)
            self.dem = raw_dem

            class _B:
                def __init__(self, left, bottom, right, top):
                    self.left, self.bottom, self.right, self.top = left, bottom, right, top
            b = _B(b_left, b_bottom, b_right, b_top)

            if src.crs is not None and src.crs.is_geographic:
                mean_lat = (b.bottom + b.top) / 2.0
                mean_lon = (b.left + b.right) / 2.0
                utm_zone = int((mean_lon + 180) / 6) + 1
                utm_epsg = f"EPSG:326{utm_zone:02d}" if mean_lat >= 0 else f"EPSG:327{utm_zone:02d}"
                to_utm = Transformer.from_crs(src.crs, utm_epsg, always_xy=True)

                lon_mesh, lat_mesh = np.meshgrid(xs_raw, ys_raw)
                x_utm, y_utm = to_utm.transform(lon_mesh, lat_mesh)
                mid_row, mid_col = x_utm.shape[0] // 2, x_utm.shape[1] // 2
                self.xs = x_utm[mid_row, :]
                self.ys = y_utm[:, mid_col]
                self.utm_epsg = utm_epsg
            else:
                self.xs = xs_raw
                self.ys = ys_raw
                self.utm_epsg = None

            self.off_x, self.off_y = np.mean(self.xs), np.mean(self.ys)
            self.bounds = b

        self.texture = Image.open(texture_path) if texture_path else None
        self.mesh = self._make_mesh()

        self.scene = pyrender.Scene(ambient_light=[0.15, 0.15, 0.15])
        self.scene.add(pyrender.Mesh.from_trimesh(self.mesh))

        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=4.0)
        light_pose = self._look_dir_pose(np.array([-0.4, 0.5, -0.75]))
        self.scene.add(light, pose=light_pose)

        self.renderer = pyrender.OffscreenRenderer(int(render_width), int(render_height))

    @staticmethod
    def _look_dir_pose(direction):
        d = np.asarray(direction, dtype=np.float64)
        d = d / np.linalg.norm(d)
        up_guess = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.99 else np.array([0.0, 1.0, 0.0])
        right = np.cross(up_guess, d)
        right = right / np.linalg.norm(right)
        up = np.cross(d, right)
        pose = np.eye(4)
        pose[:3, 0], pose[:3, 1], pose[:3, 2], pose[:3, 3] = right, up, d, [0, 0, 0]
        return pose

    def get_extent(self):
        return {
            "x_min": float(self.xs.min()),
            "x_max": float(self.xs.max()),
            "y_min": float(self.ys.min()),
            "y_max": float(self.ys.max()),
        }

    def _make_mesh(self):
        gx, gy = np.meshgrid(self.xs - self.off_x, self.ys - self.off_y)
        verts = np.column_stack([gx.ravel(), gy.ravel(), self.dem.ravel()])

        h, w = self.dem.shape
        idx = np.arange(h * w).reshape(h, w)
        v0, v1, v2, v3 = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
        faces = np.vstack([np.column_stack([v0, v1, v2]), np.column_stack([v1, v3, v2])])

        if self.texture is None:
            return trimesh.Trimesh(vertices=verts, faces=faces, process=False)

        u = (gx.ravel() - gx.min()) / (gx.max() - gx.min())
        v = (gy.ravel() - gy.min()) / (gy.max() - gy.min())
        uvs = np.column_stack([u, v])
        visual = trimesh.visual.TextureVisuals(uv=uvs, image=self.texture)
        return trimesh.Trimesh(vertices=verts, faces=faces, visual=visual, process=False)

    def _to_local_xy(self, x, y):
        if self.utm_epsg is None:
            return float(x), float(y)
        to_utm = Transformer.from_crs(self.dem_crs, self.utm_epsg, always_xy=True)
        ux, uy = to_utm.transform(x, y)
        return float(ux), float(uy)

    def _sample_elevation(self, x_dem, y_dem):
        """Bilinear DEM interpolation."""
        x_frac = (x_dem - self.bounds.left) / (self.bounds.right - self.bounds.left) * (self.dem.shape[1] - 1)
        y_frac = (y_dem - self.bounds.bottom) / (self.bounds.top - self.bounds.bottom) * (self.dem.shape[0] - 1)
        
        col0 = int(np.clip(np.floor(x_frac), 0, self.dem.shape[1] - 2))
        col1 = col0 + 1
        row0 = int(np.clip(np.floor(y_frac), 0, self.dem.shape[0] - 2))
        row1 = row0 + 1
        
        tx = x_frac - col0
        ty = y_frac - row0
        
        z00 = self.dem[row0, col0]
        z10 = self.dem[row0, col1]
        z01 = self.dem[row1, col0]
        z11 = self.dem[row1, col1]
        
        return float((1.0 - tx) * (1.0 - ty) * z00 + tx * (1.0 - ty) * z10 + (1.0 - tx) * ty * z01 + tx * ty * z11)

    def get_render(self, cam_xy, target_xy, yfov=np.pi / 3):
        target_z = self._sample_elevation(target_xy[0], target_xy[1])
        cam_x_local, cam_y_local = self._to_local_xy(cam_xy[0], cam_xy[1])
        tgt_x_local, tgt_y_local = self._to_local_xy(target_xy[0], target_xy[1])

        eye = np.array([cam_x_local - self.off_x, cam_y_local - self.off_y, target_z + self.camera_height_m])
        at = np.array([tgt_x_local - self.off_x, tgt_y_local - self.off_y, target_z])
        return self._render_from(eye, at, yfov)

    def get_render_first_person(self, eye_xy, azimuth_deg, pitch_deg=0.0, eye_height_m=1.6, yfov=np.pi / 3,
                                 look_distance_m=5000.0):
        # Clamp camera coordinate to be safely inside the active grid boundaries
        # (completely prevents standing outside the terrain mesh boundaries)
        x_safe = np.clip(eye_xy[0], self.xs[0] + 10.0, self.xs[-1] - 10.0)
        y_safe = np.clip(eye_xy[1], self.ys[0] + 10.0, self.ys[-1] - 10.0)

        ground_z = self._sample_elevation(x_safe, y_safe)
        eye_z = ground_z + float(eye_height_m)
        eye_x_local, eye_y_local = self._to_local_xy(x_safe, y_safe)

        az = np.deg2rad(azimuth_deg)
        pt = np.deg2rad(pitch_deg)
        
        look_dx = np.sin(az) * np.cos(pt) * look_distance_m
        look_dy = np.cos(az) * np.cos(pt) * look_distance_m
        look_dz = np.sin(pt) * look_distance_m

        eye = np.array([eye_x_local - self.off_x, eye_y_local - self.off_y, eye_z])
        at = np.array([eye_x_local - self.off_x + look_dx, eye_y_local - self.off_y + look_dy, eye_z + look_dz])
        return self._render_from(eye, at, yfov)

    def _render_from(self, eye, at, yfov):
        forward = (at - eye) / np.linalg.norm(at - eye)
        right = np.cross(forward, [0, 0, 1])
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-8:
            right = np.array([1.0, 0.0, 0.0])
            right_norm = 1.0
        up = np.cross(right / np.linalg.norm(right), forward)

        pose = np.eye(4)
        pose[:3, 0], pose[:3, 1], pose[:3, 2], pose[:3, 3] = right / right_norm, up, -forward, eye
        c_node = self.scene.add(pyrender.PerspectiveCamera(yfov=float(yfov)), pose=pose)
        img, _ = self.renderer.render(self.scene)
        self.scene.remove_node(c_node)
        return img

    def close(self):
        if getattr(self, "renderer", None) is not None:
            self.renderer.delete()
            self.renderer = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass