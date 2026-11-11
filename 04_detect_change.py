"""Step 4 — predict forest for BEFORE and AFTER years and flag newly cleared areas.

A pixel is 'cleared' if:
  * the model was confident it was forest before (p > P_FOREST_BEFORE),
  * confident it is not forest after (p < P_FOREST_AFTER),
  * NDVI dropped by at least NDVI_DROP (guards against model noise), and
  * both years had a clear (cloud-free) observation.
Then small speckles are removed and patches are turned into map polygons.
"""
import json

import numpy as np
import rasterio
import torch
from rasterio.features import shapes
from rasterio.warp import transform_geom
from scipy import ndimage

import config as C
from common import UNet, add_indices, predict_image, read_tif, valid_mask

norm = np.load(C.DATA / "norm.npz")
mean, std = norm["mean"], norm["std"]
model = UNet(in_ch=len(mean), base=C.BASE_CHANNELS)
model.load_state_dict(torch.load(C.OUT / "unet.pt", map_location="cpu"))

probs, ndvis, valids = {}, {}, {}
for year in (C.BEFORE_YEAR, C.AFTER_YEAR):
    img, profile = read_tif(C.composite_path(year))
    feats = add_indices(img)
    probs[year] = predict_image(model, feats, mean, std)
    ndvis[year] = feats[6]
    valids[year] = valid_mask(img)
    prof = profile.copy(); prof.update(count=1, dtype="float32", nodata=None)
    with rasterio.open(C.OUT / f"forest_prob_{year}.tif", "w", **prof) as dst:
        dst.write(probs[year][None])
    print(f"{year}: forest cover {(probs[year][valids[year]] > 0.5).mean():.1%} of clear pixels")

b, a = C.BEFORE_YEAR, C.AFTER_YEAR
both_valid = valids[b] & valids[a]
cleared = ((probs[b] > C.P_FOREST_BEFORE) & (probs[a] < C.P_FOREST_AFTER)
           & ((ndvis[b] - ndvis[a]) > C.NDVI_DROP) & both_valid)

# Clean up: morphological opening removes 1-2 pixel noise; then drop tiny patches
cleared = ndimage.binary_opening(cleared, structure=np.ones((3, 3)))
lab, n = ndimage.label(cleared)
pixel_ha = (C.RESOLUTION ** 2) / 10_000
sizes = ndimage.sum(cleared, lab, range(1, n + 1)) * pixel_ha
keep = np.isin(lab, np.where(sizes >= C.MIN_AREA_HA)[0] + 1)
cleared = keep

prof = profile.copy(); prof.update(count=1, dtype="uint8", nodata=None)
with rasterio.open(C.OUT / "cleared_mask.tif", "w", **prof) as dst:
    dst.write(cleared.astype("uint8")[None])

# Vectorise -> GeoJSON polygons (lat/lon) with area per patch
lab, n = ndimage.label(cleared)
features = []
for geom, val in shapes(lab.astype("int32"), mask=cleared, transform=profile["transform"]):
    area = float((lab == int(val)).sum() * pixel_ha)
    features.append({
        "type": "Feature",
        "geometry": transform_geom(profile["crs"], "EPSG:4326", geom),
        "properties": {"id": int(val), "area_ha": round(area, 2)},
    })
json.dump({"type": "FeatureCollection", "features": features}, open(C.OUT / "cleared_areas.geojson", "w"))

forest_before = (probs[b] > 0.5) & both_valid
stats = {
    "area_km2": round(both_valid.sum() * pixel_ha / 100, 1),
    f"forest_{b}_ha": round(float(forest_before.sum() * pixel_ha)),
    f"forest_{a}_ha": round(float(((probs[a] > 0.5) & both_valid).sum() * pixel_ha)),
    "cleared_ha": round(float(cleared.sum() * pixel_ha), 1),
    "cleared_pct_of_forest": round(100 * cleared.sum() / max(forest_before.sum(), 1), 2),
    "n_patches": int(n),
    "largest_patch_ha": round(max([f["properties"]["area_ha"] for f in features], default=0), 1),
}
json.dump(stats, open(C.OUT / "change_stats.json", "w"), indent=2)
print(json.dumps(stats, indent=2))
