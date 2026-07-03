import os
import json
import numpy as np
import rasterio
from pyproj import Transformer
from PIL import Image
import pyrender
import trimesh
import math

# Headless rendering setup
os.environ['PYRENDER_BACKEND'] = 'egl'

DATASET_DIR = "data/synthetic_dataset"
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
MASKS_DIR = os.path.join(DATASET_DIR, "masks")
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(MASKS_DIR, exist_ok=True)

TOTAL_SAMPLES = 50

print("Loading DEM and detecting georeferencing...")
with rasterio.open("data/dem.tif") as src:
    dem_data = src.read(1).astype(np.float32)
    [pixel_width, row_rotation, start_x, col_rotation, pixel_height, start_y] = src.transform[:6]

    raw_xs = start_x + np.arange(src.width) * pixel_width
    raw_ys = start_y + np.arange(src.height) * pixel_height
    
    if pixel_height < 0:
        raw_ys = raw_ys[::-1]
        dem_data = np.flipud(dem_data)
        
    dem_crs = src.crs.to_string()

center_x = raw_xs[len(raw_xs) // 2]
center_y = raw_ys[len(raw_ys) // 2]

crop_size_x, crop_size_y = 110000.0, 110000.0
crop_min_x, crop_max_x = center_x - crop_size_x/2.0, center_x + crop_size_x/2.0
crop_min_y, crop_max_y = center_y - crop_size_y/2.0, center_y + crop_size_y/2.0

crop_mask_x = (raw_xs >= crop_min_x) & (raw_xs <= crop_max_x)
crop_mask_y = (raw_ys >= crop_min_y) & (raw_ys <= crop_max_y)

xs_cropped = raw_xs[crop_mask_x]
ys_cropped = raw_ys[crop_mask_y]
dem_data_cropped = dem_data[crop_mask_y][:, crop_mask_x]

stride = 2  
dem_data_final = dem_data_cropped[::stride, ::stride]
xs_final = xs_cropped[::stride]
ys_final = ys_cropped[::stride]
dem_height, dem_width = dem_data_final.shape

vertices = np.column_stack((np.meshgrid(xs_final.astype(np.float32), ys_final.astype(np.float32))[0].ravel(),
                            np.meshgrid(xs_final.astype(np.float32), ys_final.astype(np.float32))[1].ravel(),
                            dem_data_final.ravel()))

r, c = np.meshgrid(np.arange(dem_height - 1, dtype=np.int32), np.arange(dem_width - 1, dtype=np.int32), indexing='ij')
v0 = r * dem_width + c
v1, v2, v3 = v0 + 1, v0 + dem_width, v0 + dem_width + 1
faces = np.vstack((np.column_stack((v0.ravel(), v1.ravel(), v2.ravel())),
                    np.column_stack((v1.ravel(), v3.ravel(), v2.ravel()))))

dem_to_gps = Transformer.from_crs(dem_crs, "EPSG:4326", always_xy=True)
gps_min_lon, gps_min_lat = dem_to_gps.transform(xs_final[0], ys_final[0])
gps_max_lon, gps_max_lat = dem_to_gps.transform(xs_final[-1], ys_final[-1])

sat_image = Image.open("data/satellite_texture_cropped.png")
bounds_data = np.load("data/satellite_texture_bounds_cropped.npz")
img_min_lat = float(bounds_data['min_lat'])
img_max_lat = float(bounds_data['max_lat'])
img_min_lon = float(bounds_data['min_lon'])
img_max_lon = float(bounds_data['max_lon'])

sat_array = np.array(sat_image)
stitched_h, stitched_w, _ = sat_array.shape

lon_v, lat_v = dem_to_gps.transform(vertices[:, 0], vertices[:, 1])
u = np.clip((lon_v - img_min_lon) / (img_max_lon - img_min_lon), 0.0, 1.0)
v = np.clip((lat_v - img_min_lat) / (img_max_lat - img_min_lat), 0.0, 1.0)

pixel_x = (u * (stitched_w - 1)).astype(np.int32)
pixel_y = ((1.0 - v) * (stitched_h - 1)).astype(np.int32)
sampled_colors = sat_array[pixel_y, pixel_x]

dy, dx = np.gradient(dem_data_final, 30.0, 30.0)
slope_magnitude = np.sqrt(dx**2 + dy**2)
slope_2d = np.arctan(slope_magnitude) / (np.pi / 2.0)
up_similarity = 1.0 - slope_2d.ravel()
ambient_occlusion = np.clip(0.20 + 0.80 * (up_similarity.ravel() ** 1.8), 0.15, 1.0)

vertex_colors = np.zeros((vertices.shape[0], 4), dtype=np.uint8)
vertex_colors[:, :3] = np.clip(sampled_colors * ambient_occlusion[:, np.newaxis], 0, 255).astype(np.uint8)
vertex_colors[:, 3] = 255

terrain_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False, validate=False)
terrain_mesh.visual.vertex_colors = vertex_colors
terrain_mesh.vertex_normals = np.column_stack((-dx.ravel()/np.sqrt(dx**2+dy**2+0.01).ravel(), -dy.ravel()/np.sqrt(dx**2+dy**2+0.01).ravel(), np.ones_like(dx).ravel()/np.sqrt(dx**2+dy**2+0.01).ravel())).astype(np.float32)
pyrender_mesh = pyrender.Mesh.from_trimesh(terrain_mesh, smooth=True)

img_w, img_h = 2160, 1440
np.random.seed(42)
l1 = np.array(Image.fromarray((np.random.uniform(0, 1, (8, 12)) * 255).astype(np.uint8)).resize((img_w, img_h), Image.Resampling.BILINEAR)) / 255.0
l2 = np.array(Image.fromarray((np.random.uniform(0, 1, (24, 36)) * 255).astype(np.uint8)).resize((img_w, img_h), Image.Resampling.BILINEAR)) / 255.0
l3 = np.array(Image.fromarray((np.random.uniform(0, 1, (72, 108)) * 255).astype(np.uint8)).resize((img_w, img_h), Image.Resampling.BILINEAR)) / 255.0

peak_indices = np.where(dem_data_final > 6000.0)
if len(peak_indices[0]) == 0:
    peak_indices = np.where(dem_data_final > 5000.0)

def generate_sample_render(sample_id, config):
    scene = pyrender.Scene(ambient_light=config["ambient_light"])
    scene.add(pyrender_mesh)
    
    sun_light = pyrender.DirectionalLight(color=config["sun_color"], intensity=config["sun_intensity"])
    scene.add(sun_light, pose=config["sun_pose"])
    
    camera = pyrender.PerspectiveCamera(yfov=np.radians(config["fov_y_deg"]), aspectRatio=1.5, znear=1.0, zfar=15000.0)
    scene.add(camera, pose=config["cam_pose"])
    
    renderer = pyrender.OffscreenRenderer(img_w, img_h)
    raw_color, depth = renderer.render(scene)
    renderer.delete()
    
    sky_mask = depth == 0.0
    sky_ratio = np.sum(sky_mask) / sky_mask.size
    if sky_ratio < 0.15 or sky_ratio > 0.65:
        raise ValueError("Skyline composition out of bounds.")
    
    y_idx_sky, x_idx_sky = np.indices((img_h, img_w))
    y_factor = (y_idx_sky / img_h)[:, :, np.newaxis]
    sky_background = (1.0 - y_factor**1.2) * config["sky_top"] + (y_factor**1.2) * config["sky_bottom"]
    
    dist_to_sun = np.sqrt((x_idx_sky - config["sun_img_x"])**2 + (y_idx_sky - config["sun_img_y"])**2)
    sun_glow = np.exp(-dist_to_sun / 90.0)[:, :, np.newaxis]
    sky_background += sun_glow * config["sun_glow_color"] * 0.90
    
    cloud_noise = l1 * 0.55 + l2 * 0.32 + l3 * 0.13
    cloud_density = np.clip((cloud_noise - 0.44) * 3.2, 0.0, 1.0) * np.sin((y_idx_sky / img_h) * np.pi) * config["cloud_coverage"]
    cloud_density = cloud_density[:, :, np.newaxis]
    
    sky_final = (1.0 - cloud_density) * sky_background + cloud_density * config["cloud_color"]
    sky_background_final = np.clip(sky_final, 0, 255).astype(np.uint8)
    
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
    
    final_canvas = Image.fromarray(camera_array_grain)
    final_image_resized = final_canvas.resize((1080, 720), resample=Image.Resampling.LANCZOS)
    
    binary_mask = (sky_mask * 255).astype(np.uint8)
    mask_canvas = Image.fromarray(binary_mask)
    final_mask_resized = mask_canvas.resize((1080, 720), resample=Image.Resampling.NEAREST)
    
    return final_image_resized, final_mask_resized

print("Beginning procedural generation loop...")

viewpoints_mapping_path = "data/viewpoints_mapping.npy"
if os.path.exists(viewpoints_mapping_path):
    viewpoints_mapping = np.load(viewpoints_mapping_path)
else:
    from pyproj import Transformer
    gps_to_utm_fb = Transformer.from_crs("EPSG:4326", "EPSG:32645", always_xy=True)
    utm_to_gps_fb = Transformer.from_crs("EPSG:32645", "EPSG:4326", always_xy=True)
    min_x_fb, min_y_fb = gps_to_utm_fb.transform(86.582, 27.770)
    max_x_fb, max_y_fb = gps_to_utm_fb.transform(86.989, 28.041)
    X_v_fb = np.arange(min_x_fb, max_x_fb, 500.0)
    Y_v_fb = np.arange(min_y_fb, max_y_fb, 500.0)
    view_xs_fb, view_ys_fb = np.meshgrid(X_v_fb, Y_v_fb, indexing='xy')
    flat_xs_fb = view_xs_fb.ravel(order='C')  
    flat_ys_fb = view_ys_fb.ravel(order='C')
    lons_fb, lats_fb = utm_to_gps_fb.transform(flat_xs_fb, flat_ys_fb)
    viewpoints_mapping = np.column_stack((lats_fb, lons_fb, flat_xs_fb, flat_ys_fb)).astype(np.float32)

metadata_dict = {}
viewpoint_step = max(1, viewpoints_mapping.shape[0] // TOTAL_SAMPLES)

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
        if attempts > 100:
            print(f"Error: Could not find valid view for sample {sample_id} after 100 attempts.")
            break

        eye_x = float(viewpoints_mapping[target_vp_idx, 2])
        eye_y = float(viewpoints_mapping[target_vp_idx, 3])
        ix = int(np.argmin(np.abs(xs_final - eye_x)))
        iy = int(np.argmin(np.abs(ys_final - eye_y)))
        ix = np.clip(ix, 0, dem_width - 1)
        iy = np.clip(iy, 0, dem_height - 1)

        ground_z = dem_data_final[iy, ix]
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
        
        weather = rng.choice(["sunny", "overcast", "sunset", "stormy_haze"])
        sun_azim = rng.uniform(-np.pi, np.pi)
        sun_elev = rng.uniform(np.radians(35.0), np.radians(65.0))
        
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
            "cam_pose": cam_pose, "sun_pose": sun_pose,
            "grain_intensity": rng.uniform(1.0, 2.5),
            "fov_y_deg": rng.uniform(55.0, 75.0)
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
            
        config["sun_img_x"] = int(img_w * rng.uniform(0.20, 0.80))
        config["sun_img_y"] = int(img_h * rng.uniform(0.10, 0.35))
        
        try:
            final_img, final_mask = generate_sample_render(sample_id, config)
            final_img.save(img_path)
            final_mask.save(os.path.join(MASKS_DIR, img_filename))
            
            true_lon, true_lat = dem_to_gps.transform(eye_x, eye_y)
            dists = np.sum((viewpoints_mapping[:, 2:4] - np.array([eye_x, eye_y]))**2, axis=1)
            closest_idx = int(np.argmin(dists))
            closest_dist = np.sqrt(dists[closest_idx])
            
            true_heading_deg = math.degrees(math.atan2(forward_norm[0], forward_norm[1])) % 360.0
            
            # --- CRITICAL: COMPUTE TILT ROTATION MATRIX R_tilt ---
            R = cam_pose[:3, :3]
            forward_vec = -R[:, 2]
            forward_horiz_plane = np.array([forward_vec[0], forward_vec[1], 0.0], dtype=np.float32)
            forward_horiz_len = np.linalg.norm(forward_horiz_plane)
            
            if forward_horiz_len < 1e-5:
                # Fallback to identity matrix if camera looks straight up or down
                R_tilt = np.eye(3, dtype=np.float32)
            else:
                forward_horiz_plane /= forward_horiz_len
                # Construct leveled world coordinate frame aligned with current heading
                z_level = -forward_horiz_plane  # looks down -Z
                y_level = np.array([0.0, 0.0, 1.0], dtype=np.float32)  # looks up +Y
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
                "cam_R_tilt": R_tilt.tolist()  # Save tilt correction matrix
            }
            render_successful = True
            print(f"✓ Sample {sample_id} rendered successfully on attempt {attempts}.")
        except ValueError:
            pass
        except Exception as e:
            import traceback
            print(f"\n[CRITICAL ERROR] during rendering: {e}")
            traceback.print_exc()
            import sys
            sys.exit(1)

gt_json_path = "data/synthetic_dataset_gt.json"
with open(gt_json_path, "w") as f:
    json.dump(metadata_dict, f, indent=4)

print("Procedural generation completed.")
