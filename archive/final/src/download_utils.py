import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

import earthaccess
import requests
from pyprojroot import here

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from PIL import Image
import io

from src.region import import_region


def load_download_bounds(bounds_json=None):
    if bounds_json is None:
        bounds_json = here() / "notebooks" / "01_RegionStudy" / "output" / "download_bounds.json"
    region = import_region(bounds_json)
    return region, [region.west_deg, region.south_deg, region.east_deg, region.north_deg]


def login_earthdata():
    auth = earthaccess.login()
    if not auth.authenticated:
        raise RuntimeError("Earthaccess login failed")
    return auth


def search_data_by_bbox(short_name, bounds):
    return earthaccess.search_data(short_name=short_name, bounding_box=tuple(bounds))


def download_results_parallel(results, download_dir=None, max_workers=8):
    """Download granules via earthaccess-managed auth/session handling."""
    if download_dir is None:
        download_dir = here() / "data" / "digital_elevation_model" / "download"

    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)

    # Remove zero-byte leftovers so downloader can fetch them again.
    for tif in download_dir.glob("*.tif"):
        try:
            if tif.stat().st_size == 0:
                tif.unlink()
        except OSError:
            pass

    # earthaccess handles Earthdata auth/session and retries for protected endpoints.
    try:
        files = earthaccess.download(results, local_path=str(download_dir), threads=max_workers)
    except TypeError:
        # Backward-compat for earthaccess versions without `threads`.
        files = earthaccess.download(results, local_path=str(download_dir))

    files = files or []
    return {"success": len(files), "failed": max(0, len(results) - len(files)), "total": len(results), "download_dir": str(download_dir)}


def _download_one(url, download_dir, username, password, timeout):
    filename = Path(urlparse(url).path).name or "download.bin"
    dst = download_dir / filename
    local_size = dst.stat().st_size if dst.exists() else 0
    headers = {"Range": f"bytes={local_size}-"} if local_size > 0 else {}

    with requests.get(
        url,
        stream=True,
        auth=(username, password),
        headers=headers,
        timeout=timeout,
    ) as r:
        if r.status_code == 206 and local_size > 0:
            mode = "ab"
        else:
            mode = "wb"
        r.raise_for_status()
        with open(dst, mode) as out:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    out.write(chunk)
    return filename


def download_urls_parallel(urls_file=None, download_dir=None, max_workers=8, timeout=(10, 60)):
    if urls_file is None:
        urls_file = here() / "data" / "digital_elevation_model" / "urls.txt"
    if download_dir is None:
        download_dir = here() / "data" / "digital_elevation_model" / "download"

    urls_file = Path(urls_file)
    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)

    username = os.getenv("EARTHDATA_USERNAME")
    password = os.getenv("EARTHDATA_PASSWORD")
    if not username or not password:
        raise ValueError("EARTHDATA_USERNAME and EARTHDATA_PASSWORD must be set in .env")

    with open(urls_file, "r") as f:
        urls = [line.strip() for line in f if line.strip()]

    workers = min(max_workers, max(1, len(urls)))
    ok = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_download_one, url, download_dir, username, password, timeout) for url in urls]
        for future in as_completed(futures):
            try:
                print("Downloaded:", future.result())
                ok += 1
            except Exception as e:
                print("Failed:", e)
                failed += 1

    return {"success": ok, "failed": failed, "total": len(urls), "download_dir": str(download_dir)}

def download_esri_satellite(region, output_img_path, output_bounds_path, width=2048, height=2048):
    """
    Downloads regional satellite imagery from the Esri World Imagery REST API
    using a robust session with built-in exponential backoff.
    """
    # 1. Setup standard exponential backoff retry strategy
    session = requests.Session()
    retry_strategy = Retry(
        total=5,
        backoff_factor=2,  # Backoff delays: 2s, 4s, 8s, 16s...
        status_forcelist=[429, 500, 502, 503, 504],
    )
    session.mount("https://", HTTPAdapter(max_retries=retry_strategy))

    # Esri Export Image REST API parameters
    url = "https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/export"
    params = {
        "bbox": f"{region.west_deg},{region.south_deg},{region.east_deg},{region.north_deg}",
        "bboxSR": "4326",
        "imageSR": "4326",
        "size": f"{width},{height}",
        "format": "png",
        "f": "image",
    }

    print(f"Querying Esri World Imagery API with exponential backoff...")
    response = session.get(url, params=params, timeout=30)
    response.raise_for_status()

    # Save the imagery
    img = Image.open(io.BytesIO(response.content))
    img.save(output_img_path)
    print(f"✓ Saved satellite texture to {output_img_path}")

    # Save matching metadata bounds for the 3D scene builder
    np.savez(
        output_bounds_path,
        min_lon=region.west_deg,
        max_lon=region.east_deg,
        min_lat=region.south_deg,
        max_lat=region.north_deg,
    )
    print(f"✓ Saved texture bounds mapping to {output_bounds_path}")