"""Step 1 — download cloud-free Sentinel-2 composites + ESA WorldCover labels.

Data comes from Microsoft Planetary Computer (free, no account needed).
For each year we take every dry-season scene, mask clouds/shadows with the
Sentinel-2 scene classification layer (SCL), and take the per-pixel median.
"""
import functools
import os
import time

print = functools.partial(print, flush=True)  # show messages immediately
print("Step 1: loading libraries (first run can take ~30 s on Windows)...")

# Make satellite downloads more robust on home Wi-Fi: retry failed reads instead of hanging
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "5")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "2")
os.environ.setdefault("GDAL_HTTP_TIMEOUT", "60")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")

import dask
import numpy as np
import planetary_computer
import pystac_client
import rasterio
from dask.diagnostics import ProgressBar
from odc.stac import load

import config as C

print("Connecting to Microsoft Planetary Computer...")
CATALOG = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)
print("Connected.")
# SCL classes to keep: 4 vegetation, 5 bare soil, 6 water, 7 unclassified
CLEAR_SCL = [4, 5, 6, 7]


def save_tif(path, arr, geobox, nodata=None):
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = arr[None]
    with rasterio.open(
        path, "w", driver="GTiff", height=arr.shape[1], width=arr.shape[2],
        count=arr.shape[0], dtype=arr.dtype, crs=C.CRS,
        transform=geobox.affine, nodata=nodata, compress="deflate",
    ) as dst:
        dst.write(arr)


def composite(year, geobox=None):
    start, end = (f"{year}-{d}" for d in C.SEASON)
    print(f"\n{year}: searching for Sentinel-2 scenes {start} to {end}...")
    t0 = time.time()
    items = CATALOG.search(
        collections=["sentinel-2-l2a"], bbox=C.BBOX, datetime=f"{start}/{end}",
        query={"eo:cloud_cover": {"lt": C.MAX_CLOUD}},
    ).item_collection()
    items = sorted(items, key=lambda i: i.properties["eo:cloud_cover"])[: C.MAX_SCENES]
    if not items:
        raise SystemExit(f"No scenes found for {year}; raise MAX_CLOUD in config.py")
    print(f"{year}: {len(items)} scenes, cloud cover "
          f"{[round(i.properties['eo:cloud_cover']) for i in items]}")

    # All years share one pixel grid so they can be compared pixel-by-pixel.
    grid = {"geobox": geobox} if geobox is not None else \
        {"bbox": C.BBOX, "crs": C.CRS, "resolution": C.RESOLUTION}
    ds = load(items, bands=C.BANDS + ["SCL"], groupby="solar_day", **grid,
              chunks={"x": 1024, "y": 1024}, resampling={"SCL": "nearest", "*": "bilinear"})

    # Since processing baseline 04.00 (Jan 2022) ESA adds +1000 to reflectance values.
    # Remove it per scene so 2019 and 2024 are on the same scale.
    offset_by_day = {}
    for it in items:
        day = it.datetime.date()
        baseline = float(it.properties.get("s2:processing_baseline", "0") or 0)
        offset_by_day[day] = 1000 if baseline >= 4.0 else 0
    offsets = np.array([offset_by_day.get(t.astype("datetime64[D]").item(), 0)
                        for t in ds.time.values], dtype="float32")

    clear = ds["SCL"].isin(CLEAR_SCL)
    bands = []
    for b in C.BANDS:
        refl = (ds[b].astype("float32") - offsets[:, None, None]) / 10000.0
        refl = refl.where(clear & (ds[b] > 0)).clip(0, 1)
        # median needs the whole time axis in one dask chunk
        bands.append(refl.chunk({"time": -1}).median("time", skipna=True))
    print(f"{year}: downloading + compositing (slow part, watch the progress bar)...")
    with ProgressBar(dt=2):
        stack = np.stack(dask.compute(*[b.data for b in bands])).astype("float32")
    print(f"{year}: finished in {(time.time() - t0) / 60:.1f} min")
    valid = np.isfinite(stack).all(0)
    print(f"{year}: {valid.mean():.1%} of pixels have a clear observation")
    stack = np.nan_to_num(stack, nan=0.0)
    save_tif(C.composite_path(year), stack, ds.odc.geobox, nodata=0)
    return ds.odc.geobox


def labels(geobox):
    print(f"\nDownloading ESA WorldCover {C.TRAIN_YEAR} labels...")
    items = CATALOG.search(collections=["esa-worldcover"], bbox=C.BBOX,
                           datetime=f"{C.TRAIN_YEAR}").item_collection()
    wc = load(items, bands=["map"], geobox=geobox, resampling="nearest")["map"]
    wc = wc.max("time").values.astype("uint8")   # tiles don't overlap; max merges them
    print("WorldCover classes present:", {int(k): int(v) for k, v in
                                          zip(*np.unique(wc, return_counts=True))})
    forest = (wc == 10).astype("uint8")          # class 10 = tree cover
    save_tif(C.LABEL_PATH, forest, geobox, nodata=None)
    print(f"Forest fraction in labels: {forest.mean():.1%}")


if __name__ == "__main__":
    geobox = composite(C.TRAIN_YEAR)
    labels(geobox)
    for y in (C.BEFORE_YEAR, C.AFTER_YEAR):
        composite(y, geobox)
    print("Done ->", C.DATA)
