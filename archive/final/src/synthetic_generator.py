"""Generates high-fidelity synthetic mountain views with advanced procedural weather."""
import os
import json
import math
import numpy as np
import cv2
import rasterio
from pyproj import Transformer
from PIL import Image
import trimesh
import pyrender
from tqdm import tqdm

# Headless Pyrender EGL configuration
os.environ["PYRENDER_BACKEND"] = "egl"


class SyntheticSceneGenerator:
    def __init__(self, dem_path, sat_path, bounds_path, clouds_dir=None, stride=4):
        """Loads DEM and satellite imagery to build the high-fidelity 3D terrain mesh."""
        self.clouds_dir = clouds_dir
        self.cloud_files = []
        if self.clouds_dir and os.path.exists(self.clouds_dir):
            self.cloud_files = sorted([
                os.path.join(self.clouds_dir, f) 
                for f in os.listdir(self.clouds_dir) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ])
            print(f"✓ Found {len(self.cloud_files)} cloud images for backdrops.")

        print("Loading DEM and detecting georeferencing...")
        with rasterio.open(dem_path) as src:
            dem_data = src.read(1).astype(np.float32)
            dem_data = np.nan_to_num(dem_data, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Fill empty borders
            valid_mask = dem_data > 10.0
            if np.any(valid_mask):
                min_valid = float(np.min(dem_data[valid_mask]))
                dem_data[~valid_mask] = min_valid
            else:
                dem_data[~valid_mask] = 1000.0
            dem_data = np.clip(dem_data, 100.0, 9000.0)

            pixel_width, _, start_x, _, pixel_height, start_y = src.transform[:6]
            self.xs = start_x + np.arange(src.width) * pixel_width
            self.ys = start_y + np.arange(src.height) * pixel_height
            if pixel_height < 0:
                self.ys = self.ys[::-1]
                dem_data = np.flipud(dem_data)
            self.dem_crs = src.crs.to_string()

        # Decimate and smooth DEM
        self.dem_data = dem_data[::stride, ::stride]
        self.dem_data = cv2.medianBlur(self.dem_data, 5)
        self.xs = self.xs[::stride]
        self.ys = self.ys[::stride]
        self.dem_height, self.dem_width = self.dem_data.shape

        # Offset coordinates to prevent single-precision float issues
        self.center_x_offset = float(np.mean(self.xs))
        self.center_y_offset = float(np.mean(self.ys))

        gx, gy = np.meshgrid(self.xs - self.center_x_offset, self.ys - self.center_y_offset)
        verts = np.column_stack([gx.ravel(), gy.ravel(), self.dem_data.ravel()])

        idx = np.arange(self.dem_height * self.dem_width).reshape(self.dem_height, self.dem_width)
        v0, v1, v2, v3 = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
        faces = np.vstack([np.column_stack([v0, v1, v2]), np.column_stack([v1, v3, v2])])

        # Map satellite texture coordinates
        to_gps = Transformer.from_crs(self.dem_crs, "EPSG:4326", always_xy=True)
        lon_v, lat_v = to_gps.transform(gx.ravel() + self.center_x_offset, gy.ravel() + self.center_y_offset)
        
        bounds_data = np.load(bounds_path)
        img_min_lat = float(bounds_data['min_lat'])
        img_max_lat = float(bounds_data['max_lat'])
        img_min_lon = float(bounds_data['min_lon'])
        img_max_lon = float(bounds_data['max_lon'])

        lon_range = max(1e-5, img_max_lon - img_min_lon)
        lat_range = max(1e-5, img_max_lat - img_min_lat)
        u = np.clip((lon_v - img_min_lon) / lon_range, 0.0, 1.0)
        v = np.clip((lat_v - img_min_lat) / lat_range, 0.0, 1.0)
        uvs = np.column_stack((u, v))

        # Build primary terrain mesh
        sat_image = Image.open(sat_path)
        terrain_mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False, validate=False)
        terrain_mesh.visual = trimesh.visual.TextureVisuals(uv=uvs, image=sat_image)
        self.pyrender_mesh = pyrender.Mesh.from_trimesh(terrain_mesh, smooth=True)

        # Build vertical perimeter wall (skirt) to prevent side voids
        perimeter_indices = []
        for c in range(self.dem_width): perimeter_indices.append(idx[0, c])
        for r in range(1, self.dem_height): perimeter_indices.append(idx[r, self.dem_width - 1])
        for c in range(self.dem_width - 2, -1, -1): perimeter_indices.append(idx[self.dem_height - 1, c])
        for r in range(self.dem_height - 2, 0, -1): perimeter_indices.append(idx[r, 0])

        P = len(perimeter_indices)
        z_base = np.min(self.dem_data) - 1000.0  # Base level below lowest point

        skirt_verts = np.zeros((2 * P, 3), dtype=np.float32)
        skirt_verts[:P] = verts[perimeter_indices]
        skirt_verts[P:] = verts[perimeter_indices]
        skirt_verts[P:, 2] = z_base

        skirt_uvs = np.zeros((2 * P, 2), dtype=np.float32)
        skirt_uvs[:P] = uvs[perimeter_indices]
        skirt_uvs[P:] = uvs[perimeter_indices]

        skirt_faces = []
        for i in range(P):
            tl = i
            tr = (i + 1) % P
            bl = P + i
            br = P + ((i + 1) % p_len if 'p_len' in locals() else (i + 1) % P)
            skirt_faces.extend([[tl, bl, tr], [tr, bl, br], [tl, tr, bl], [tr, br, bl]])

        skirt_mesh = trimesh.Trimesh(vertices=skirt_verts, faces=np.array(skirt_faces), process=False, validate=False)
        skirt_mesh.visual = trimesh.visual.TextureVisuals(uv=skirt_uvs, image=sat_image)
        self.pyrender_skirt_mesh = pyrender.Mesh.from_trimesh(skirt_mesh, smooth=True)

        # Persistent offscreen renderer to prevent RAM leaks
        self.img_w, self.img_h = 2160, 1440
        self.renderer = pyrender.OffscreenRenderer(self.img_w, self.img_h)
        self.dem_to_gps = to_gps

    def _sample_elevation_bilinear(self, x, y):
        """Bilinear height lookup to prevent camera burial on steep slopes."""
        dx = self.xs[1] - self.xs[0]
        dy = self.ys[1] - self.ys[0]
        
        i_frac = (x - self.xs[0]) / dx
        j_frac = (y - self.ys[0]) / dy
        
        i0 = int(np.clip(np.floor(i_frac), 0, self.dem_width - 2))
        i1 = i0 + 1
        j0 = int(np.clip(np.floor(j_frac), 0, self.dem_height - 2))
        j1 = j0 + 1
        
        tx = np.clip(i_frac - i0, 0.0, 1.0)
        ty = np.clip(j_frac - j0, 0.0, 1.0)
        
        z00 = self.dem_data[j0, i0]
        z10 = self.dem_data[j0, i1]
        z01 = self.dem_data[j1, i0]
        z11 = self.dem_data[j1, i1]
        
        return float((1.0 - tx) * (1.0 - ty) * z00 + tx * (1.0 - ty) * z10 + (1.0 - tx) * ty * z01 + tx * ty * z11)

    def generate_backdrop_view(self, sample_id, config, depth, sky_mask, raw_color):
        """Generates visual elements: sky gradients, stars, sun flares, rain, and sensor grain."""
        y_idx_sky, x_idx_sky = np.indices((self.img_h, self.img_w))
        rng = np.random.default_rng(sample_id)

        # 1. Base sky gradient or cloud photo
        sky_rendered = False
        if self.cloud_files and not config["is_night"]:
            try:
                cloud_path = self.cloud_files[sample_id % len(self.cloud_files)]
                with Image.open(cloud_path) as c_img:
                    c_img_resized = c_img.resize((self.img_w, self.img_h), Image.Resampling.LANCZOS)
                    sky_base = np.array(c_img_resized.convert("RGB")).astype(np.float32)
                    sky_rendered = True
            except Exception:
                sky_rendered = False
                
        if not sky_rendered:
            y_factor = (y_idx_sky / self.img_h)[:, :, np.newaxis]
            sky_base = (1.0 - y_factor**1.2) * config["sky_top"] + (y_factor**1.2) * config["sky_bottom"]
            
            if config["is_night"]:
                num_stars = 250
                star_x = rng.integers(0, self.img_w, size=num_stars)
                star_y = rng.integers(0, self.img_h, size=num_stars)
                star_brightness = rng.uniform(0.6, 1.0, size=num_stars)
                for i in range(num_stars):
                    sb = int(star_brightness[i] * 255.0)
                    cv2.circle(sky_base, (star_x[i], star_y[i]), 1, (sb, sb, sb + 30), -1)

        # 2. Render procedural sun/moon spot
        dist_to_sun = np.sqrt((x_idx_sky - config["sun_img_x"])**2 + (y_idx_sky - config["sun_img_y"])**2)
        sun_core = np.exp(-dist_to_sun / 15.0)[:, :, np.newaxis] * 255.0
        sun_glow = np.exp(-dist_to_sun / 110.0)[:, :, np.newaxis]
        
        sky_final = sky_base + sun_core + sun_glow * config["sun_glow_color"] * 1.25
        
        # 3. Advanced lens effects (diffraction rings & bokeh particles)
        if config["has_sun_effects"]:
            sun_cx, sun_cy = config["sun_img_x"], config["sun_img_y"]
            img_cx, img_cy = self.img_w / 2.0, self.img_h / 2.0
            vec_x, vec_y = img_cx - sun_cx, img_cy - sun_cy
            
            # Secondary Lens Flare Rings
            for offset in [0.35, 0.70, -0.25]:
                flare_x = int(sun_cx + vec_x * offset)
                flare_y = int(sun_cy + vec_y * offset)
                dist_to_flare = np.sqrt((x_idx_sky - flare_x)**2 + (y_idx_sky - flare_y)**2)
                ring = np.exp(-(dist_to_flare - 80.0)**2 / 600.0)[:, :, np.newaxis]
                sky_final += ring * config["sun_glow_color"] * 0.18
                
            # Golden Bokeh Dust Motes
            p_color = np.array([255.0, 252.0, 242.0])
            for _ in range(30):
                p_dist = rng.uniform(20.0, 400.0)
                p_angle = rng.uniform(0, 2*np.pi)
                px = int(sun_cx + p_dist * np.cos(p_angle))
                py = int(sun_cy + p_dist * np.sin(p_angle))
                if 10 <= px < self.img_w - 10 and 10 <= py < self.img_h - 10:
                    pr = int(rng.uniform(2, 5))
                    alpha = (1.0 - p_dist / 400.0) * rng.uniform(0.3, 0.6)
                    
                    roi = sky_final[py-pr:py+pr+1, px-pr:px+pr+1]
                    y_g, x_g = np.ogrid[-pr:pr+1, -pr:pr+1]
                    mask = (x_g**2 + y_g**2 <= pr**2)[:, :, np.newaxis]
                    
                    sky_final[py-pr:py+pr+1, px-pr:px+pr+1] = np.where(
                        mask,
                        (1.0 - alpha) * roi + alpha * p_color,
                        roi
                    )

        # 4. Horizon Haze & Extinction Blending
        horizon_blend = np.clip((y_idx_sky / self.img_h) ** 1.8, 0.0, 1.0)[:, :, np.newaxis]
        sky_background_final = (1.0 - horizon_blend) * sky_final + horizon_blend * config["fog_color"]
        sky_background_final = np.clip(sky_background_final, 0, 255).astype(np.uint8)
        
        extinction = np.exp(-config["fog_density"] * depth)
        extinction = np.clip(extinction, 0.0, 1.0)[:, :, np.newaxis]
        
        processed_image = (1.0 - extinction) * config["fog_color"] + extinction * raw_color.astype(np.float32)
        processed_image[sky_mask] = sky_background_final[sky_mask]
        processed_image = np.clip(processed_image, 0, 255).astype(np.uint8)
        
        # 5. Chromatic Aberration Post-Processing
        img_pil = Image.fromarray(processed_image)
        r_chan = img_pil.getchannel('R')
        r_w, r_h = int(self.img_w * 1.0015), int(self.img_h * 1.0015)
        r_resized = r_chan.resize((r_w, r_h), Image.Resampling.LANCZOS)
        left_r, top_r = (r_w - self.img_w) // 2, (r_h - self.img_h) // 2
        r_final = r_resized.crop((left_r, top_r, left_r + self.img_w, top_r + self.img_h))
        g_final = img_pil.getchannel('G')
        
        b_chan = img_pil.getchannel('B')
        b_w, b_h = int(self.img_w * 0.9985), int(self.img_h * 0.9985)
        b_resized = b_chan.resize((b_w, b_h), Image.Resampling.LANCZOS)
        b_final = Image.new('L', (self.img_w, self.img_h), 0)
        left_b, top_b = (self.img_w - b_w) // 2, (self.img_h - b_h) // 2
        b_final.paste(b_resized, (left_b, top_b))
        
        camera_processed = Image.merge('RGB', (r_final, g_final, b_final))
        camera_array = np.array(camera_processed).astype(np.float32)
        
        # 6. Vignette Mask
        cy, cx = self.img_h / 2.0, self.img_w / 2.0
        y_grid, x_grid = np.meshgrid(np.arange(self.img_h), np.arange(self.img_w), indexing='ij')
        norm_dist_to_center = np.sqrt(((x_grid - cx) / cx)**2 + ((y_grid - cy) / cy)**2)
        vignette_mask = np.clip(1.0 - 0.35 * (norm_dist_to_center ** 2), 0.0, 1.0)[:, :, np.newaxis]
        camera_array *= vignette_mask
        
        # 7. Sensor Noise Grain
        sensor_grain = rng.normal(0, config["grain_intensity"], size=camera_array.shape).astype(np.float32)
        camera_array_grain = np.clip(camera_array + sensor_grain, 0, 255).astype(np.uint8)

        # 8. Rain filter falling rain streaks
        if config["is_rainy"]:
            num_streaks = 2000
            rx = rng.integers(0, self.img_w, size=num_streaks)
            ry = rng.integers(0, self.img_h, size=num_streaks)
            for i in range(num_streaks):
                px, py = int(rx[i]), int(ry[i])
                length = int(rng.uniform(15, 35))
                thickness = int(rng.uniform(1, 2))
                end_x = int(px - length * 0.15)
                end_y = int(py + length)
                
                if 0 <= px < self.img_w and 0 <= py < self.img_h and 0 <= end_x < self.img_w and 0 <= end_y < self.img_h:
                    cv2.line(camera_array_grain, (px, py), (end_x, end_y), (185, 195, 205), thickness, lineType=cv2.LINE_AA)

        return camera_array_grain

    def generate_dataset(self, viewpoints, target_samples, images_dir, masks_dir, gt_json_path):
        """Runs the loop with restored weather simulations."""
        images_dir = Path(images_dir)
        masks_dir = Path(masks_dir)
        images_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)

        gt_dict = {}
        samples_completed = 0
        
        peak_indices = np.where(self.dem_data > 6000.0)
        if len(peak_indices[0]) == 0:
            peak_indices = np.where(self.dem_data > 5000.0)

        # Wrap in tqdm
        pbar = tqdm(total=target_samples, desc="Generating Synthetic Weather Views")
        
        for idx in range(len(viewpoints)):
            if samples_completed >= target_samples:
                break

            rng = np.random.default_rng(idx)
            
            # Seeding randomized spatial offsets to prevent grid cheating
            offset_x, offset_y = rng.uniform(-15.0, 15.0, size=2)
            offset_x = np.sign(offset_x) * max(1.0, abs(offset_x))
            offset_y = np.sign(offset_y) * max(1.0, abs(offset_y))

            eye_x = float(viewpoints[idx, 2]) + offset_x
            eye_y = float(viewpoints[idx, 3]) + offset_y

            # Slope edge clamping
            dist_to_edge_x = min(eye_x - self.xs[0], self.xs[-1] - eye_x)
            dist_to_edge_y = min(eye_y - self.ys[0], self.ys[-1] - eye_y)
            if dist_to_edge_x < 8000.0 or dist_to_edge_y < 8000.0:
                continue

            ground_z = self._sample_elevation_bilinear(eye_x, eye_y)
            eye_z = ground_z + 1.8
            eye = np.array([eye_x, eye_y, eye_z], dtype=np.float32)

            # Pick a peak to look at
            peak_match_idx = rng.integers(0, len(peak_indices[0]))
            p_ix = peak_indices[1][peak_match_idx]
            p_iy = peak_indices[0][peak_match_idx]
            peak_x, peak_y, peak_z = self.xs[p_ix], self.ys[p_iy], self.dem_data[p_iy, p_ix]
            
            dir_to_peak = np.array([peak_x - eye[0], peak_y - eye[1], peak_z - eye[2]], dtype=np.float32)
            dir_to_peak /= np.linalg.norm(dir_to_peak)
            target = eye + dir_to_peak * 1000.0

            # Orient Camera Pose
            forward = target - eye
            forward_norm = forward / np.linalg.norm(forward)
            forward_horiz = np.array([forward_norm[0], forward_norm[1], 0.0])
            forward_horiz /= np.linalg.norm(forward_horiz)
            up_global = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            right = np.cross(forward_horiz, up_global)
            right /= np.linalg.norm(right)
            up_camera = np.cross(right, forward_norm)
            up_camera /= np.linalg.norm(up_camera)
            
            cam_pose = np.eye(4, dtype=np.float32)
            cam_pose[:3, 0] = right
            cam_pose[:3, 1] = up_camera
            cam_pose[:3, 2] = -forward_norm
            cam_pose[:3, 3] = eye
            
            cam_pose_render = cam_pose.copy()
            cam_pose_render[0, 3] -= self.center_x_offset
            cam_pose_render[1, 3] -= self.center_y_offset

            # Procedural Weather Selector
            weather = rng.choice(["sunny", "overcast", "sunset", "stormy_haze", "night", "rainy"])
            sun_azim = rng.uniform(-np.pi, np.pi)
            sun_elev = rng.uniform(np.radians(15.0), np.radians(65.0))
                  
            z_axis = np.array([np.cos(sun_elev)*np.sin(sun_azim), np.cos(sun_elev)*np.cos(sun_azim), np.sin(sun_elev)], dtype=np.float32)
            z_axis /= np.linalg.norm(z_axis)
            x_axis = np.cross([0, 0, 1], z_axis)
            if np.linalg.norm(x_axis) < 1e-5: x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            else: x_axis /= np.linalg.norm(x_axis)
            y_axis = np.cross(z_axis, x_axis)
            y_axis /= np.linalg.norm(y_axis)
            
            sun_pose = np.eye(4, dtype=np.float32)
            sun_pose[:3, 0] = x_axis
            sun_pose[:3, 1] = y_axis
            sun_pose[:3, 2] = z_axis
                 
            config = {
                "cam_pose_render": cam_pose_render, 
                "sun_pose": sun_pose,
                "grain_intensity": rng.uniform(1.0, 2.5),
                "fov_y_deg": rng.uniform(55.0, 75.0),
                "is_night": False,
                "is_rainy": False,
                "has_sun_effects": False
            }
            
            if weather == "sunny":
                config["ambient_light"] = [0.45, 0.50, 0.65]
                config["sun_color"] = [1.0, 0.98, 0.92]
                config["sun_intensity"] = rng.uniform(3.5, 4.2)
                config["sky_top"] = np.array([12.0, 65.0, 190.0])
                config["sky_bottom"] = np.array([120.0, 185.0, 245.0])
                config["sun_glow_color"] = np.array([255.0, 245.0, 215.0])
                config["cloud_coverage"] = rng.uniform(0.0, 0.25)
                config["cloud_color"] = np.array([250.0, 252.0, 255.0])
                config["fog_density"] = rng.uniform(0.000001, 0.000003)
                config["fog_color"] = np.array([175, 195, 220], dtype=np.float32)
                config["has_sun_effects"] = True if rng.uniform() > 0.3 else False
            elif weather == "overcast":
                config["ambient_light"] = [0.35, 0.35, 0.38]
                config["sun_color"] = [0.95, 0.95, 0.95]
                config["sun_intensity"] = rng.uniform(1.2, 1.8)
                config["sky_top"] = np.array([110.0, 115.0, 125.0])
                config["sky_bottom"] = np.array([180.0, 185.0, 190.0])
                config["sun_glow_color"] = np.array([200.0, 200.0, 200.0])
                config["cloud_coverage"] = rng.uniform(0.65, 0.95)
                config["cloud_color"] = np.array([225.0, 227.0, 230.0])
                config["fog_density"] = rng.uniform(0.000015, 0.000030)
                config["fog_color"] = np.array([190, 195, 205], dtype=np.float32)
            elif weather == "sunset":
                config["ambient_light"] = [0.20, 0.15, 0.30]
                config["sun_color"] = [1.0, 0.40, 0.10]
                config["sun_intensity"] = rng.uniform(2.5, 3.2)
                config["sky_top"] = np.array([30.0, 20.0, 75.0])
                config["sky_bottom"] = np.array([255.0, 100.0, 30.0])
                config["sun_glow_color"] = np.array([255.0, 150.0, 60.0])
                config["cloud_coverage"] = rng.uniform(0.15, 0.65)
                config["cloud_color"] = np.array([240.0, 115.0, 95.0])
                config["fog_density"] = rng.uniform(0.000006, 0.000015)
                config["fog_color"] = np.array([220, 135, 115], dtype=np.float32)
                config["has_sun_effects"] = True if rng.uniform() > 0.3 else False
            elif weather == "stormy_haze":
                config["ambient_light"] = [0.25, 0.26, 0.28]
                config["sun_color"] = [0.95, 0.90, 0.85]
                config["sun_intensity"] = rng.uniform(1.5, 2.2)
                config["sky_top"] = np.array([65.0, 65.0, 70.0])
                config["sky_bottom"] = np.array([140.0, 140.0, 145.0])
                config["sun_glow_color"] = np.array([160.0, 155.0, 145.0])
                config["cloud_coverage"] = rng.uniform(0.40, 0.85)
                config["cloud_color"] = np.array([150.0, 150.0, 155.0])
                config["fog_density"] = rng.uniform(0.000010, 0.000025)
                config["fog_color"] = np.array([155, 160, 165], dtype=np.float32)
            elif weather == "night":
                config["ambient_light"] = [0.03, 0.04, 0.08]
                config["sun_color"] = [0.85, 0.90, 0.95]
                config["sun_intensity"] = rng.uniform(0.3, 0.6)
                config["sky_top"] = np.array([3.0, 4.0, 12.0])
                config["sky_bottom"] = np.array([10.0, 12.0, 22.0])
                config["sun_glow_color"] = np.array([150.0, 180.0, 210.0])
                config["cloud_coverage"] = rng.uniform(0.0, 0.40)
                config["cloud_color"] = np.array([150.0, 180.0, 210.0])
                config["fog_density"] = rng.uniform(0.000001, 0.000008)
                config["fog_color"] = np.array([10, 12, 25], dtype=np.float32)
                config["is_night"] = True
            elif weather == "rainy":
                config["ambient_light"] = [0.22, 0.22, 0.25]
                config["sun_color"] = [0.80, 0.82, 0.85]
                config["sun_intensity"] = rng.uniform(0.6, 1.2)
                config["sky_top"] = np.array([50.0, 52.0, 58.0])
                config["sky_bottom"] = np.array([100.0, 102.0, 108.0])
                config["sun_glow_color"] = np.array([120.0, 122.0, 128.0])
                config["cloud_coverage"] = rng.uniform(0.75, 1.0)
                config["cloud_color"] = np.array([90.0, 92.0, 98.0])
                config["fog_density"] = rng.uniform(0.000020, 0.000045)
                config["fog_color"] = np.array([110, 112, 118], dtype=np.float32)
                config["is_rainy"] = True

            if rng.uniform(0.0, 1.0) < 0.30:
                config["fog_density"] *= rng.uniform(2.5, 4.5)            

            config["sun_img_x"] = int(self.img_w * rng.uniform(0.20, 0.80))
            config["sun_img_y"] = int(self.img_h * rng.uniform(0.10, 0.35))

            try:
                # EGL Render
                scene = pyrender.Scene(ambient_light=config["ambient_light"])
                scene.add(self.pyrender_mesh)
                scene.add(self.pyrender_skirt_mesh)

                sun_light = pyrender.DirectionalLight(color=config["sun_color"], intensity=config["sun_intensity"])
                scene.add(sun_light, pose=config["sun_pose"])
                
                camera = pyrender.PerspectiveCamera(yfov=np.radians(config["fov_y_deg"]), aspectRatio=1.5, znear=5.0, zfar=80000.0)
                scene.add(camera, pose=config["cam_pose_render"])
                
                # Reuse offscreen renderer to prevent memory leak
                raw_color, depth = self.renderer.render(scene)
                sky_mask = depth == 0.0

                # Composition Validation
                if np.mean(sky_mask[0, :]) < 0.97 or np.any(sky_mask[-1, :]):
                    raise ValueError("Blocked view")
                    
                terrain_depths = depth[~sky_mask]
                if len(terrain_depths) > 0 and np.min(terrain_depths) < 300.0:
                    raise ValueError("Foreground blockage")

                sky_ratio = np.sum(sky_mask) / sky_mask.size
                if sky_ratio < 0.15 or sky_ratio > 0.65:
                    raise ValueError("Unbalanced sky ratio")

                # Build Procedural Weather Backdrop
                rendered_img_array = self.generate_backdrop_view(idx, config, depth, sky_mask, raw_color)
                
                # Invert mask: Sky = 255 (White), Terrain = 0 (Black)
                sky_binary = np.where(sky_mask, 0, 255).astype(np.uint8)
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(sky_binary, connectivity=8)
                valid_sky_regions = 0
                for i in range(1, num_labels):
                    if stats[i, cv2.CC_STAT_AREA] > 100:  
                        valid_sky_regions += 1
                if valid_sky_regions >= 2:
                    raise ValueError("Multi-sky void region detected")

                img_filename = f"sample_{samples_completed:04d}.png"
                
                # Save processed high-res camera view downscaled to 1080p
                final_canvas = Image.fromarray(rendered_img_array)
                final_canvas.resize((1080, 720), resample=Image.Resampling.LANCZOS).save(images_dir / img_filename)

                # Save mask downscaled to 1080p
                mask_canvas = Image.fromarray(sky_binary)
                mask_canvas.resize((1080, 720), resample=Image.Resampling.NEAREST).save(masks_dir / img_filename)

                # Record exact metadata
                true_lon, true_lat = self.dem_to_gps.transform(eye_x, eye_y)
                
                # Calculate True Heading
                dir_to_peak = np.array([peak_x - eye[0], peak_y - eye[1], peak_z - eye[2]], dtype=np.float32)
                dir_to_peak /= np.linalg.norm(dir_to_peak)
                true_heading_deg = math.degrees(math.atan2(dir_to_peak[0], dir_to_peak[1])) % 360.0
                
                # Camera Tilt Extraction
                R = cam_pose[:3, :3]
                forward_vec = -R[:, 2]
                forward_horiz_plane = np.array([forward_vec[0], forward_vec[1], 0.0], dtype=np.float32)
                forward_horiz_len = np.linalg.norm(forward_horiz_plane)
                if forward_horiz_len < 1e-5:
                    R_tilt = np.eye(3, dtype=np.float32)
                else:
                    forward_horiz_plane /= forward_horiz_len
                    z_level = -forward_horiz_plane
                    y_level = np.array([0.0, 0.0, 1.0], dtype=np.float32)
                    x_level = np.cross(y_level, z_level)
                    x_level /= np.linalg.norm(x_level)
                    R_level_frame = np.column_stack((x_level, y_level, z_level))
                    R_tilt = R_level_frame.T @ R

                gt_dict[str(samples_completed)] = {
                    "true_lat": float(true_lat), 
                    "true_lon": float(true_lon),
                    "eye_x_utm": float(eye_x), 
                    "eye_y_utm": float(eye_y), 
                    "eye_z_m": float(eye_z),
                    "fov_y_deg": float(config["fov_y_deg"]),
                    "true_heading_deg": float(true_heading_deg),
                    "cam_R_tilt": R_tilt.tolist()
                }

                samples_completed += 1
                pbar.update(1)
            except ValueError:
                continue

        pbar.close()
        self.renderer.delete()
        
        with open(gt_json_path, "w") as f:
            json.dump(gt_dict, f, indent=4)
        print(f"✓ Saved ground truth JSON metadata to {gt_json_path}")