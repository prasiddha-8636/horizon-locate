"""Equirectangular panorama → perspective crop utilities.

Corrects a bug in the prior cropper that placed sky at the bottom of the
perspective output by inverting the Y axis projection.
"""

import numpy as np
from PIL import Image


def vertical_to_horizontal_fov(vertical_fov_deg, aspect_ratio):
    """Vertical FOV → horizontal FOV via thin-lens model."""
    return np.degrees(
        2.0 * np.arctan(np.tan(np.radians(vertical_fov_deg) / 2.0) * aspect_ratio)
    )


def slice_perspective(
    pano_path,
    heading_deg,
    pitch_deg=0.0,
    roll_deg=0.0,
    fov_y_deg=65.0,
    out_w=1080,
    out_h=720,
):
    """Extract a perspective crop from a cylindrical equirectangular pano.

    Convention used here:
    - pano column 0 maps to longitude 0 deg (East)
    - pano column increases East → West → East... (i.e. heading 0 = col 0)
    - pano row 0 is +90 deg latitude (zenith, looking up)
    - pano row H-1 is -90 deg latitude (nadir, looking down)

    The camera coordinate frame is:
    - +X right, +Y up, -Z forward (camera looks down -Z)

    A ray in world frame (X=East, Y=Up, Z=North) corresponds to:
        lon = atan2(X, -Z)   in radians, range (-pi, pi]
        lat = asin(Y / r)     in radians, range [-pi/2, pi/2]
        pano_col = (lon / (2*pi)) * pano_w   wrapped into [0, pano_w)
        pano_row = (0.5 - lat/pi) * pano_h  (top of pano = +lat = +90 = zenith)
    """
    pano_img = Image.open(pano_path).convert("RGB")
    pano_w, pano_h = pano_img.size
    pano_arr = np.asarray(pano_img)

    aspect = out_w / out_h
    fov_y_rad = np.radians(fov_y_deg)
    fov_x_rad = np.radians(vertical_to_horizontal_fov(fov_y_deg, aspect))

    # Pixel coords in output
    u, v = np.meshgrid(np.arange(out_w), np.arange(out_h))
    # Camera ray directions (in camera frame, +X right, +Y up, -Z forward)
    # tan(half_fov) = (half_extent) / focal
    # x = (u - cx) / focal_x  where cx = out_w/2
    x_cam = (u - out_w / 2.0) * np.tan(fov_x_rad / 2.0) / (out_w / 2.0)
    y_cam = -(v - out_h / 2.0) * np.tan(fov_y_rad / 2.0) / (out_h / 2.0)  # +Y up
    z_cam = -np.ones_like(x_cam)
    rays = np.stack([x_cam, y_cam, z_cam], axis=-1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)

    # Apply camera rotation (yaw / pitch / roll) to convert to world frame.
    # World frame: X=East, Y=Up, Z=North. Compass heading h: forward = (sin h, cos h)
    h = np.radians(heading_deg)
    p = np.radians(pitch_deg)
    r = np.radians(roll_deg)

    # Yaw: maps camera -Z (forward) to world (sin h, cos h) and camera +X
    # (right) to the camera's right side (compass h+90). Fixes the prior
    # 180-degree heading error and the horizontal mirror.
    cy, sy = np.cos(h), np.sin(h)
    R_yaw = np.array(
        [
            [cy, 0.0, -sy],
            [0.0, 1.0, 0.0],
            [-sy, 0.0, -cy],
        ]
    )
    # Pitch (rotation about world X, positive = look up)
    cp, sp = np.cos(p), np.sin(p)
    R_pitch = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cp, -sp],
            [0.0, sp, cp],
        ]
    )
    # Roll (rotation about world Z)
    cr, sr = np.cos(r), np.sin(r)
    R_roll = np.array(
        [
            [cr, -sr, 0.0],
            [sr, cr, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    R_world = R_yaw @ R_pitch @ R_roll
    rays_world = rays @ R_world.T  # row-vector convention

    # Map world rays to pano pixel coords. Google pano columns run North
    # (col 0) increasing clockwise (col w/4 = East), so the column index for a
    # world ray is (compass azimuth / 360) * w. lon is measured from South;
    # compass azimuth = 180 - degrees(lon) in this frame, giving:
    #   col = (0.5 - lon/(2*pi)) * w
    lon = np.arctan2(rays_world[..., 0], -rays_world[..., 2])  # (-pi, pi]
    lat = np.arcsin(np.clip(rays_world[..., 1], -1.0, 1.0))  # [-pi/2, pi/2]

    col_f = (0.5 - lon / (2.0 * np.pi)) * pano_w
    row_f = (0.5 - lat / np.pi) * pano_h  # top of pano = +90 deg = zenith

    # Wrap columns (seamless)
    col_f = np.mod(col_f, pano_w)
    # Clamp rows (avoid sampling outside pano vertically)
    row_f = np.clip(row_f, 0, pano_h - 1.001)

    # Bilinear sample
    x0 = np.floor(col_f).astype(np.int32)
    y0 = np.floor(row_f).astype(np.int32)
    x1 = (x0 + 1) % pano_w
    y1 = np.minimum(y0 + 1, pano_h - 1)
    wa = (col_f - x0)[..., None]
    wb = 1.0 - wa
    wc = (row_f - y0)[..., None]
    wd = 1.0 - wc

    out = (
        wa * wc * pano_arr[y0, x0]
        + wb * wc * pano_arr[y0, x1]
        + wa * wd * pano_arr[y1, x0]
        + wb * wd * pano_arr[y1, x1]
    ).astype(np.uint8)

    return Image.fromarray(out)


def build_tilt_matrix(pitch_rad, roll_rad):
    """Compose pitch (about X) and roll (about Z) into a 3x3 matrix."""
    cp, sp = np.cos(pitch_rad), np.sin(pitch_rad)
    R_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cp, -sp],
            [0.0, sp, cp],
        ]
    )
    cr, sr = np.cos(roll_rad), np.sin(roll_rad)
    R_z = np.array(
        [
            [cr, -sr, 0.0],
            [sr, cr, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return (R_x @ R_z).tolist()
