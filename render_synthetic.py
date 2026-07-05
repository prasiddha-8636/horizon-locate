import os
import json
import numpy as np
import rasterio
from pyproj import Transformer
from PIL import Image
import pyrender
import trimesh
import math
import cv2
import time

os.environ['PYRENDER_BACKEND'] = 'egl'

DATASET_DIR = "data/synthetic_dataset"
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
MASKS_DIR = os.path.join(DATASET_DIR, "masks")
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(MASKS_DIR, exist_ok=True)

CLOUDS_DIR = "data/clouds"
cloud_files = []
if os.path.exists(CLOUDS_DIR):
    cloud_files = sorted([
        os.path.join(CLOUDS_DIR, f) 
        for f in os.listdir(CLOUDS_DIR) 
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    print(f"✓ Found {len(cloud_files)} real cloud images for backdrops.")
else:
    print(f"Warning: {CLOUDS_DIR} directory not found.")

TOTAL_SAMPLES = 300

print("Loading DEM and detecting georeferencing...")
with rasterio.open("data/digital_elevation_model/dem.tif") as src:
    dem_data = src.read(1).astype(np.float32)
    
    # Clean any invalid NaN/Inf values
    dem_data = np.nan_to_num(dem_data, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Identify valid mountain elevations (> 10.0m) and fill empty borders with the mountain-base level
    valid_mask = dem_data > 10.0
    if np.any(valid_mask):
        min_valid = float(np.min(dem_data[valid_mask]))
        dem_data[~valid_mask] = min_valid
    else:
        dem_data[~valid_mask] = 1000.0
        
    dem_data = np.clip(dem_data, 100.0, 9000.0)

    [pixel_width, row_rotation, start_x, col_rotation, pixel_height, start_y] = src.transform[:6]

    raw_xs = start_x + np.arange(src.width) * pixel_width
    raw_ys = start_y + np.arange(src.height) * pixel_height
    
    if pixel_height < 0:
        raw_ys = raw_ys[::-1]
        dem_data = np.flipud(dem_data)
        
    dem_crs = src.crs.to_string()

viewpoints_mapping_path = "data/digital_elevation_model/viewpoints_mapping.npy"
if os.path.exists(viewpoints_mapping_path):
    viewpoints_mapping = np.load(viewpoints_mapping_path)
    center_x = float(np.mean(viewpoints_mapping[:, 2]))
    center_y = float(np.mean(viewpoints_mapping[:, 3]))
else:
    center_x = 478712.0
    center_y = 3086932.0

crop_size_x, crop_size_y = 200000.0, 190000.0

# Extract 1D boolean masks indicating which columns/rows contain valid mountain data
valid_cols = np.any(valid_mask, axis=0)
valid_rows = np.any(valid_mask, axis=1)

crop_min_x = max(center_x - crop_size_x/2.0, np.min(raw_xs[valid_cols]))
crop_max_x = min(center_x + crop_size_x/2.0, np.max(raw_xs[valid_cols]))
crop_min_y = max(center_y - crop_size_y/2.0, np.min(raw_ys[valid_rows]))
crop_max_y = min(center_y + crop_size_y/2.0, np.max(raw_ys[valid_rows]))

crop_mask_x = (raw_xs >= crop_min_x) & (raw_xs <= crop_max_x)
crop_mask_y = (raw_ys >= crop_min_y) & (raw_ys <= crop_max_y)

xs_cropped = raw_xs[crop_mask_x]
ys_cropped = raw_ys[crop_mask_y]
dem_data_cropped = dem_data[crop_mask_y][:, crop_mask_x]

stride = 4
dem_data_final = dem_data_cropped[::stride, ::stride]
dem_data_final = cv2.medianBlur(dem_data_final, 5)
xs_final = xs_cropped[::stride]
ys_final = ys_cropped[::stride]
dem_height, dem_width = dem_data_final.shape

if 'viewpoints_mapping' in locals() or os.path.exists(viewpoints_mapping_path):
    center_x_mesh = (xs_final[0] + xs_final[-1]) / 2.0
    center_y_mesh = (ys_final[0] + ys_final[-1]) / 2.0
    
    eye_xs = viewpoints_mapping[:, 2]
    eye_ys = viewpoints_mapping[:, 3]
    
    safe_mask = (
        (eye_xs >= center_x_mesh - 20000.0) & (eye_xs <= center_x_mesh + 20000.0) &
        (eye_ys >= center_y_mesh - 15000.0) & (eye_ys <= center_y_mesh + 15000.0)
    )
    viewpoints_mapping = viewpoints_mapping[safe_mask]
    print(f"✓ Pre-filtered viewpoints (40km x 30km region): {len(viewpoints_mapping)} coordinates.")

# Explicit grid generation (aligned to 'xy' projection)
grid_x, grid_y = np.meshgrid(xs_final.astype(np.float32), ys_final.astype(np.float32), indexing='xy')

# We make explicit contiguous copies of both the coordinate grids and the sliced elevation array
grid_x_con = np.ascontiguousarray(grid_x, dtype=np.float32)
grid_y_con = np.ascontiguousarray(grid_y, dtype=np.float32)
dem_final_con = np.ascontiguousarray(dem_data_final, dtype=np.float32)

vertices_abs = np.column_stack((
    grid_x_con.ravel(),
    grid_y_con.ravel(),
    dem_final_con.ravel()
))

# Coordinate centering to prevent floating point precision issues
center_x_offset = float(np.mean(xs_final))
center_y_offset = float(np.mean(ys_final))

vertices = vertices_abs.copy()
vertices[:, 0] -= center_x_offset
vertices[:, 1] -= center_y_offset

dem_to_gps = Transformer.from_crs(dem_crs, "EPSG:4326", always_xy=True)
gps_min_lon, gps_min_lat = dem_to_gps.transform(xs_final[0], ys_final[0])
gps_max_lon, gps_max_lat = dem_to_gps.transform(xs_final[-1], ys_final[-1])

sat_image = Image.open("data/satellite_imagery/satellite_texture_cropped.png")
bounds_data = np.load("data/satellite_imagery/satellite_texture_bounds_cropped.npz")
img_min_lat = float(bounds_data['min_lat'])
img_max_lat = float(bounds_data['max_lat'])
img_min_lon = float(bounds_data['min_lon'])
img_max_lon = float(bounds_data['max_lon'])

sat_array = np.array(sat_image)
stitched_h, stitched_w, _ = sat_array.shape

lon_v, lat_v = dem_to_gps.transform(vertices_abs[:, 0], vertices_abs[:, 1])

lon_range = max(1e-5, img_max_lon - img_min_lon)
lat_range = max(1e-5, img_max_lat - img_min_lat)
u = np.clip((lon_v - img_min_lon) / lon_range, 0.0, 1.0)
v = np.clip((lat_v - img_min_lat) / lat_range, 0.0, 1.0)

pixel_x = (u * (stitched_w - 1)).astype(np.int32)
pixel_y = ((1.0 - v) * (stitched_h - 1)).astype(np.int32)
sampled_colors = sat_array[pixel_y, pixel_x]

grid_indices = np.arange(dem_height * dem_width, dtype=np.int32).reshape(dem_height, dem_width)

# 1. Fast, robust, zero-overhead sequential grid triangulation
v0 = grid_indices[:-1, :-1].ravel()
v1 = grid_indices[:-1, 1:].ravel()
v2 = grid_indices[1:, :-1].ravel()
v3 = grid_indices[1:, 1:].ravel()
faces = np.vstack((
    np.column_stack((v0, v1, v2)),
    np.column_stack((v1, v3, v2))
))

N = vertices.shape[0]

# 2. Extract the outer perimeter indices of the grid in clockwise order
perimeter_indices = []
for c in range(dem_width):
    perimeter_indices.append(grid_indices[0, c])
for r in range(1, dem_height):
    perimeter_indices.append(grid_indices[r, dem_width - 1])
for c in range(dem_width - 2, -1, -1):
    perimeter_indices.append(grid_indices[dem_height - 1, c])
for r in range(dem_height - 2, 0, -1):
    perimeter_indices.append(grid_indices[r, 0])

P = len(perimeter_indices)
skirt_vertices = np.zeros((P, 3), dtype=np.float32)

# Position the base floor safely 1000m below the lowest valley floor
z_base = np.min(dem_data_final) - 1000.0  

# 3. Position the skirt vertices straight down to the base floor level
for i, idx in enumerate(perimeter_indices):
    v_orig = vertices[idx]
    skirt_vertices[i] = [v_orig[0], v_orig[1], z_base]

extended_vertices = np.vstack((vertices, skirt_vertices))

# 4. Triangulate the vertical walls with double-sided faces to prevent backface culling
skirt_faces = []
for i in range(P):
    v_top_l = perimeter_indices[i]
    v_top_r = perimeter_indices[(i+1)%P]
    v_bot_l = N + i
    v_bot_r = N + ((i+1)%P)
    
    # Front-facing
    skirt_faces.append([v_top_l, v_bot_l, v_top_r])
    skirt_faces.append([v_top_r, v_bot_l, v_bot_r])
    
    # Back-facing (reversed winding order)
    skirt_faces.append([v_top_l, v_top_r, v_bot_l])
    skirt_faces.append([v_top_r, v_bot_r, v_bot_l])

extended_faces = np.vstack((faces, np.array(skirt_faces, dtype=np.int32)))

# 5. Drape the satellite texture down the walls
u = np.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)

uvs = np.column_stack((u, v))
skirt_uvs = uvs[perimeter_indices]
extended_uvs = np.vstack((uvs, skirt_uvs))

terrain_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False, validate=False)
terrain_mesh.visual = trimesh.visual.TextureVisuals(uv=uvs, image=sat_image)
pyrender_mesh = pyrender.Mesh.from_trimesh(terrain_mesh, smooth=True)

# Create the separate vertical skirt mesh (Mesh 2) to block the side voids without normal-smoothing artifacts
skirt_verts = np.zeros((2 * P, 3), dtype=np.float32)
skirt_verts[:P] = vertices[perimeter_indices]
skirt_verts[P:] = vertices[perimeter_indices]
skirt_verts[P:, 2] = z_base

skirt_uvs_sep = np.zeros((2 * P, 2), dtype=np.float32)
skirt_uvs_sep[:P] = uvs[perimeter_indices]
skirt_uvs_sep[P:] = uvs[perimeter_indices]

skirt_faces_sep = []
for i in range(P):
    t_l = i
    t_r = (i + 1) % P
    b_l = P + i
    b_r = P + ((i + 1) % P)
    
    # Double-sided vertical wall faces
    skirt_faces_sep.append([t_l, b_l, t_r])
    skirt_faces_sep.append([t_r, b_l, b_r])
    skirt_faces_sep.append([t_l, t_r, b_l])
    skirt_faces_sep.append([t_r, b_r, b_l])
    
skirt_faces_sep = np.array(skirt_faces_sep, dtype=np.int32)

skirt_mesh = trimesh.Trimesh(vertices=skirt_verts, faces=skirt_faces_sep, process=False, validate=False)
skirt_mesh.visual = trimesh.visual.TextureVisuals(uv=skirt_uvs_sep, image=sat_image)
pyrender_skirt_mesh = pyrender.Mesh.from_trimesh(skirt_mesh, smooth=True)

img_w, img_h = 2160, 1440

peak_indices = np.where(dem_data_final > 6000.0)
if len(peak_indices[0]) == 0:
    peak_indices = np.where(dem_data_final > 5000.0)

def generate_sample_render(sample_id, config, renderer):
    scene = pyrender.Scene(ambient_light=config["ambient_light"])
    scene.add(pyrender_mesh)
    scene.add(pyrender_skirt_mesh)

    sun_light = pyrender.DirectionalLight(color=config["sun_color"], intensity=config["sun_intensity"])
    scene.add(sun_light, pose=config["sun_pose"])
    
    # znear/zfar ratio optimized to eliminate Z-fighting
    camera = pyrender.PerspectiveCamera(yfov=np.radians(config["fov_y_deg"]), aspectRatio=1.5, znear=5.0, zfar=80000.0)
    scene.add(camera, pose=config["cam_pose_render"])
    
    # Reuse the persistent renderer instead of recreating it
    raw_color, depth = renderer.render(scene)
    
    sky_mask = depth == 0.0

    # 1. Strictly Sky is Up: Reject if ANY terrain/skirt pixels touch the very top edge row
    if np.mean(sky_mask[0, :]) < 0.97:
    # if np.any(~sky_mask[0, :]):
        raise ValueError("Terrain/Skirt detected at the top edge of the frame.")
        
    # 2. Strictly Terrain is Down: Reject if ANY sky/void pixels appear at the bottom edge row
    if np.any(sky_mask[-1, :]):
        raise ValueError("Sky or void detected at the bottom edge of the frame.")
    
    # 3. Prevent blocky foreground: Reject if closest terrain is under 300 meters
    terrain_depths = depth[~sky_mask]
    if len(terrain_depths) > 0:
        if np.min(terrain_depths) < 300.0:
            raise ValueError("Immediate foreground terrain blockage or local triangle clipping detected.")            

    sky_ratio = np.sum(sky_mask) / sky_mask.size
    if sky_ratio < 0.15 or sky_ratio > 0.65:
        raise ValueError("Skyline composition out of bounds.")    

    y_idx_sky, x_idx_sky = np.indices((img_h, img_w))
    rng = np.random.default_rng(sample_id)

    # 1. Load and upscale a random cloud image from the data folder
    sky_rendered = False
    if cloud_files and not config["is_night"]:  # Skip cloud texture during starry nights
        try:
            cloud_path = cloud_files[sample_id % len(cloud_files)]
            with Image.open(cloud_path) as c_img:
                c_img_resized = c_img.resize((img_w, img_h), Image.Resampling.LANCZOS)
                sky_base = np.array(c_img_resized.convert("RGB")).astype(np.float32)
                sky_rendered = True
        except Exception as e:
            print(f"Warning: Failed loading cloud image {cloud_path}: {e}")
            
    if not sky_rendered:
        # High-quality fallback gradient
        y_factor = (y_idx_sky / img_h)[:, :, np.newaxis]
        sky_base = (1.0 - y_factor**1.2) * config["sky_top"] + (y_factor**1.2) * config["sky_bottom"]
        
        # STARRY NIGHT: Generate 250 crisp, twinkling stars on night backgrounds
        if config["is_night"]:
            num_stars = 250
            star_x = rng.integers(0, img_w, size=num_stars)
            star_y = rng.integers(0, img_h, size=num_stars)
            star_brightness = rng.uniform(0.6, 1.0, size=num_stars)
            for i in range(num_stars):
                sb = int(star_brightness[i] * 255.0)
                cv2.circle(sky_base, (star_x[i], star_y[i]), 1, (sb, sb, sb + 30), -1)

    # 2. Render a high-fidelity procedural sun/moon bright spot
    dist_to_sun = np.sqrt((x_idx_sky - config["sun_img_x"])**2 + (y_idx_sky - config["sun_img_y"])**2)
    sun_core = np.exp(-dist_to_sun / 15.0)[:, :, np.newaxis] * 255.0
    sun_glow = np.exp(-dist_to_sun / 110.0)[:, :, np.newaxis]
    
    sky_final = sky_base + sun_core + sun_glow * config["sun_glow_color"] * 1.25
    
    # 3. ADVANCED LENS EFFECTS (Flares & Glowing Dust Motes)
    if config["has_sun_effects"]:
        sun_cx, sun_cy = config["sun_img_x"], config["sun_img_y"]
        img_cx, img_cy = img_w / 2.0, img_h / 2.0
        vec_x, vec_y = img_cx - sun_cx, img_cy - sun_cy
        
        # Secondary Lens Flare Rings (diffraction rings stretching along the light vector)
        for offset in [0.35, 0.70, -0.25]:
            flare_x = int(sun_cx + vec_x * offset)
            flare_y = int(sun_cy + vec_y * offset)
            dist_to_flare = np.sqrt((x_idx_sky - flare_x)**2 + (y_idx_sky - flare_y)**2)
            ring = np.exp(-(dist_to_flare - 80.0)**2 / 600.0)[:, :, np.newaxis]
            sky_final += ring * config["sun_glow_color"] * 0.18
            
        # Glowing Sun Particles / Dust Motes floating in the sunbeams
        p_color = np.array([255.0, 252.0, 242.0]) # Warm golden-white
        for _ in range(30):
            p_dist = rng.uniform(20.0, 400.0)
            p_angle = rng.uniform(0, 2*np.pi)
            px = int(sun_cx + p_dist * np.cos(p_angle))
            py = int(sun_cy + p_dist * np.sin(p_angle))
            if 10 <= px < img_w - 10 and 10 <= py < img_h - 10:
                pr = int(rng.uniform(2, 5))
                alpha = (1.0 - p_dist / 400.0) * rng.uniform(0.3, 0.6)
                
                # Draw soft, semi-transparent bokeh particles using alpha-blending (no more black dots)
                roi = sky_final[py-pr:py+pr+1, px-pr:px+pr+1]
                y_g, x_g = np.ogrid[-pr:pr+1, -pr:pr+1]
                mask = (x_g**2 + y_g**2 <= pr**2)[:, :, np.newaxis]
                
                sky_final[py-pr:py+pr+1, px-pr:px+pr+1] = np.where(
                    mask,
                    (1.0 - alpha) * roi + alpha * p_color,
                    roi
                )

    # 4. HORIZON HAZE: Blend the bottom of the sky into the atmospheric fog
    # This dissolves any upscaling artifacts at the horizon line into the distant mountains
    horizon_blend = np.clip((y_idx_sky / img_h) ** 1.8, 0.0, 1.0)[:, :, np.newaxis]
    sky_background_final = (1.0 - horizon_blend) * sky_final + horizon_blend * config["fog_color"]
    sky_background_final = np.clip(sky_background_final, 0, 255).astype(np.uint8)
    
    extinction = np.exp(-config["fog_density"] * depth)
    extinction = np.clip(extinction, 0.0, 1.0)[:, :, np.newaxis]
    
    processed_image = (1.0 - extinction) * config["fog_color"] + extinction * raw_color.astype(np.float32)
    processed_image[sky_mask] = sky_background_final[sky_mask]
    processed_image = np.clip(processed_image, 0, 255).astype(np.uint8)
    
    img_pil = Image.fromarray(processed_image)
    r_chan = img_pil.getchannel('R')
    r_w, r_h = int(img_w * 1.0015), int(img_h * 1.0015)
    r_resized = r_chan.resize((r_w, r_h), Image.Resampling.LANCZOS)
    left_r, top_r = (r_w - img_w) // 2, (r_h - img_h) // 2
    r_final = r_resized.crop((left_r, top_r, left_r + img_w, top_r + img_h))
    g_final = img_pil.getchannel('G')
    
    b_chan = img_pil.getchannel('B')
    b_w, b_h = int(img_w * 0.9985), int(img_h * 0.9985)
    b_resized = b_chan.resize((b_w, b_h), Image.Resampling.LANCZOS)
    b_final = Image.new('L', (img_w, img_h), 0)
    left_b, top_b = (img_w - b_w) // 2, (img_h - b_h) // 2
    b_final.paste(b_resized, (left_b, top_b))
    
    camera_processed = Image.merge('RGB', (r_final, g_final, b_final))
    camera_array = np.array(camera_processed).astype(np.float32)
    
    cy, cx = img_h / 2.0, img_w / 2.0
    y_grid, x_grid = np.meshgrid(np.arange(img_h), np.arange(img_w), indexing='ij')
    norm_dist_to_center = np.sqrt(((x_grid - cx) / cx)**2 + ((y_grid - cy) / cy)**2)
    vignette_mask = np.clip(1.0 - 0.35 * (norm_dist_to_center ** 2), 0.0, 1.0)[:, :, np.newaxis]
    camera_array *= vignette_mask
    
    np.random.seed(sample_id)
    sensor_grain = np.random.normal(0, config["grain_intensity"], size=camera_array.shape).astype(np.float32)
    camera_array_grain = np.clip(camera_array + sensor_grain, 0, 255).astype(np.uint8)

    # RAIN FILTER: Draw 2,000 slanted, semi-transparent falling rain streaks
    if config["is_rainy"]:
        num_streaks = 2000
        rx = rng.integers(0, img_w, size=num_streaks)
        ry = rng.integers(0, img_h, size=num_streaks)
        
        # Soft atmospheric rain-streak drawing
        for i in range(num_streaks):
            px, py = int(rx[i]), int(ry[i])
            length = int(rng.uniform(15, 35))
            thickness = int(rng.uniform(1, 2))
            
            # Slightly angled rain direction vector
            end_x = int(px - length * 0.15)
            end_y = int(py + length)
            
            if 0 <= px < img_w and 0 <= py < img_h and 0 <= end_x < img_w and 0 <= end_y < img_h:
                # Alpha-blended soft grey-blue raindrop lines (prevents harsh cartoon lines)
                cv2.line(camera_array_grain, (px, py), (end_x, end_y), (185, 195, 205), thickness, lineType=cv2.LINE_AA)

    final_canvas = Image.fromarray(camera_array_grain)
    final_image_resized = final_canvas.resize((1080, 720), resample=Image.Resampling.LANCZOS)
    
    binary_mask = np.where(sky_mask, 0, 255).astype(np.uint8)
    mask_canvas = Image.fromarray(binary_mask)
    final_mask_resized = mask_canvas.resize((1080, 720), resample=Image.Resampling.NEAREST)
    
    return final_image_resized, final_mask_resized

print("Beginning procedural generation loop...")

metadata_dict = {}
viewpoint_step = max(1, viewpoints_mapping.shape[0] // TOTAL_SAMPLES)

renderer = pyrender.OffscreenRenderer(img_w, img_h)

loop_start_time = time.time()

for sample_id in range(TOTAL_SAMPLES):
    img_filename = f"sample_{sample_id:04d}.png"
    img_path = os.path.join(IMAGES_DIR, img_filename)
    
    if os.path.exists(img_path):
        os.remove(img_path)
    if os.path.exists(os.path.join(MASKS_DIR, img_filename)):
        os.remove(os.path.join(MASKS_DIR, img_filename))
        
    rng = np.random.default_rng(sample_id)
    render_successful = False
    attempts = 0
    target_vp_idx = (sample_id * viewpoint_step) % viewpoints_mapping.shape[0]
    
    print(f"\n--> Starting Sample {sample_id}...")
    
    while not render_successful:
        attempts += 1
        if attempts > 500:
            print(f"Error: Could not find valid view for sample {sample_id} after 500 attempts.")
            break

        eye_x = float(viewpoints_mapping[target_vp_idx, 2])
        eye_y = float(viewpoints_mapping[target_vp_idx, 3])
        
        # Skip viewpoints within 8km of the outer edges to prevent camera clipping (no more giant walls)
        dist_to_edge_x = min(eye_x - xs_final[0], xs_final[-1] - eye_x)
        dist_to_edge_y = min(eye_y - ys_final[0], ys_final[-1] - eye_y)
        if dist_to_edge_x < 8000.0 or dist_to_edge_y < 8000.0:
            target_vp_idx = (target_vp_idx + 1) % viewpoints_mapping.shape[0]
            continue

        # Bilinear interpolation of terrain height at (eye_x, eye_y)
        # to ensure the camera is placed exactly above the actual mesh surface,
        # completely preventing coordinate-mismatch clipping.
        dx = xs_final[1] - xs_final[0]
        dy = ys_final[1] - ys_final[0]
        
        i_frac = (eye_x - xs_final[0]) / dx
        j_frac = (eye_y - ys_final[0]) / dy
        
        i0 = int(np.clip(np.floor(i_frac), 0, dem_width - 2))
        i1 = i0 + 1
        j0 = int(np.clip(np.floor(j_frac), 0, dem_height - 2))
        j1 = j0 + 1
        
        tx = np.clip(i_frac - i0, 0.0, 1.0)
        ty = np.clip(j_frac - j0, 0.0, 1.0)
        
        z00 = dem_data_final[j0, i0]
        z10 = dem_data_final[j0, i1]
        z01 = dem_data_final[j1, i0]
        z11 = dem_data_final[j1, i1]
        
        ground_z = (1.0 - tx) * (1.0 - ty) * z00 + tx * (1.0 - ty) * z10 + (1.0 - tx) * ty * z01 + tx * ty * z11
        
        # Position the camera exactly 1.8m above the terrain to align perfectly with database precomputations.
        # Any nearby hillside terrain that blocks the frame will naturally trigger the sky composition checks below,
        # which safely skips the viewpoint and retries.
        eye_z = ground_z + 1.8
        
        eye = np.array([eye_x, eye_y, eye_z], dtype=np.float32)
        
        peak_match_idx = rng.integers(0, len(peak_indices[0]))
        p_ix = peak_indices[1][peak_match_idx]
        p_iy = peak_indices[0][peak_match_idx]
        
        peak_x = xs_final[p_ix]
        peak_y = ys_final[p_iy]
        peak_z = dem_data_final[p_iy, p_ix]
        
        dir_to_peak = np.array([peak_x - eye[0], peak_y - eye[1], peak_z - eye[2]], dtype=np.float32)
        dir_to_peak /= np.linalg.norm(dir_to_peak)
        target = eye + dir_to_peak * 1000.0 
        
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
        cam_pose_render[0, 3] -= center_x_offset
        cam_pose_render[1, 3] -= center_y_offset

        weather = rng.choice(["sunny", "overcast", "sunset", "stormy_haze", "night", "rainy"])
        sun_azim = rng.uniform(-np.pi, np.pi)
        sun_elev = rng.uniform(np.radians(15.0), np.radians(65.0)) # Lowered minimum elevation for better sunset/night transitions
              
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
            config["ambient_light"] = [0.03, 0.04, 0.08]  # Dim blue ambient moonlight
            config["sun_color"] = [0.85, 0.90, 0.95]  # Moon color
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
            config["ambient_light"] = [0.22, 0.22, 0.25]  # Dark cold storm grey
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

        # Dynamic valley mist / heavy fog injector (30% chance of thick atmosphere)
        if rng.uniform(0.0, 1.0) < 0.30:
            config["fog_density"] *= rng.uniform(2.5, 4.5)            

        config["sun_img_x"] = int(img_w * rng.uniform(0.20, 0.80))
        config["sun_img_y"] = int(img_h * rng.uniform(0.10, 0.35))
        
        try:
            final_img, final_mask = generate_sample_render(sample_id, config, renderer)
            
            mask_arr = np.array(final_mask) # Shape: (720, 1080)
            
            # Invert the mask: Sky becomes 255 (White), Terrain becomes 0 (Black)
            sky_binary = np.where(mask_arr == 0, 255, 0).astype(np.uint8)
            
            # Count connected sky components
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(sky_binary, connectivity=8)
            
            # Count sky regions, ignoring tiny atmospheric noise spots (< 100 pixels)
            valid_sky_regions = 0
            for i in range(1, num_labels):  # Index 0 is the background (terrain)
                area = stats[i, cv2.CC_STAT_AREA]
                if area > 100:  
                    valid_sky_regions += 1
            
            # If there are 2 or more disconnected sky regions, discard and retry
            if valid_sky_regions >= 2:
                raise ValueError("Multi-sky void region detected. Retrying viewpoint...")
            
            # Only save the file if it passes the structural validation test above
            final_img.save(img_path)
            final_mask.save(os.path.join(MASKS_DIR, img_filename))            

            true_lon, true_lat = dem_to_gps.transform(eye_x, eye_y)
            dists = np.sum((viewpoints_mapping[:, 2:4] - np.array([eye_x, eye_y]))**2, axis=1)
            closest_idx = int(np.argmin(dists))
            closest_dist = np.sqrt(dists[closest_idx])
            
            true_heading_deg = math.degrees(math.atan2(forward_norm[0], forward_norm[1])) % 360.0
            
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
            
            metadata_dict[str(sample_id)] = {
                "true_lat": float(true_lat), "true_lon": float(true_lon),
                "eye_x_utm": float(eye_x), "eye_y_utm": float(eye_y), "eye_z_m": float(eye_z),
                "closest_viewpoint_id": int(closest_idx), "closest_viewpoint_dist_m": float(closest_dist),
                "fov_y_deg": float(config["fov_y_deg"]),
                "true_heading_deg": float(true_heading_deg),
                "cam_R_tilt": R_tilt.tolist()
            }
            render_successful = True
            
            # Calculate elapsed time, average speed, and estimated remaining minutes
            elapsed_time = time.time() - loop_start_time
            avg_time_per_sample = elapsed_time / (sample_id + 1)
            est_remaining_sec = avg_time_per_sample * (TOTAL_SAMPLES - (sample_id + 1))
            est_remaining_min = est_remaining_sec / 60.0
            
            print(f"✓ Sample {sample_id} rendered successfully on attempt {attempts}.")
            print(f"  Time elapsed: {elapsed_time/60.0:.1f} min | Est. remaining: {est_remaining_min:.1f} min ({avg_time_per_sample:.1f}s/sample)")
        except ValueError:
            target_vp_idx = rng.integers(0, viewpoints_mapping.shape[0])
        except Exception as e:
            import traceback
            print(f"\n[CRITICAL ERROR] during rendering: {e}")
            traceback.print_exc()
            import sys
            sys.exit(1)

renderer.delete()

gt_json_path = "data/synthetic_dataset/ground_truth.json"
with open(gt_json_path, "w") as f:
    json.dump(metadata_dict, f, indent=4)

print("Procedural generation completed.")
