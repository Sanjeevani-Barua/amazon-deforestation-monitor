"""Step 5 — build an interactive web map (outputs/deforestation_map.html).

Layers you can toggle: true-colour imagery for both years, the model's forest
masks, and the cleared patches (click one to see its area in hectares).
"""
import base64
import io
import json

import folium
import numpy as np
import rasterio
from folium.plugins import Fullscreen, MiniMap
from PIL import Image
from rasterio.warp import Resampling, calculate_default_transform, reproject

import config as C


def to_latlon(arr, src_profile, resampling=Resampling.bilinear):
    """Reproject (bands, H, W) from UTM to EPSG:4326 so it lines up on the web map."""
    src_crs, src_tf = src_profile["crs"], src_profile["transform"]
    H, W = arr.shape[1:]
    left, top = src_tf * (0, 0); right, bottom = src_tf * (W, H)
    tf, w, h = calculate_default_transform(src_crs, "EPSG:4326", W, H, left, bottom, right, top)
    out = np.zeros((arr.shape[0], h, w), arr.dtype)
    for i in range(arr.shape[0]):
        reproject(arr[i], out[i], src_transform=src_tf, src_crs=src_crs,
                  dst_transform=tf, dst_crs="EPSG:4326", resampling=resampling)
    bounds = [[tf.f + tf.e * h, tf.c], [tf.f, tf.c + tf.a * w]]  # [[south, west], [north, east]]
    return out, bounds


def png_url(rgba):
    buf = io.BytesIO(); Image.fromarray(rgba).save(buf, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def true_colour(year):
    with rasterio.open(C.composite_path(year)) as src:
        img, prof = src.read([3, 2, 1]), src.profile   # R, G, B
    img, bounds = to_latlon(img, prof)
    rgb = np.clip(img / 0.15, 0, 1) ** (1 / 1.4)        # stretch: forest reflectance is dark
    alpha = (img.max(0) > 0) * 255
    rgba = np.dstack([(rgb.transpose(1, 2, 0) * 255).astype("uint8"), alpha.astype("uint8")])
    return png_url(rgba), bounds


def mask_layer(path, colour, thresh=0.5):
    with rasterio.open(path) as src:
        arr, prof = src.read(1)[None].astype("float32"), src.profile
    arr, bounds = to_latlon(arr, prof, Resampling.nearest)
    m = arr[0] > thresh
    rgba = np.zeros(m.shape + (4,), "uint8"); rgba[m] = colour
    return png_url(rgba), bounds


stats = json.load(open(C.OUT / "change_stats.json"))
metrics = json.load(open(C.OUT / "training_metrics.json"))
lon_c, lat_c = (C.BBOX[0] + C.BBOX[2]) / 2, (C.BBOX[1] + C.BBOX[3]) / 2
fmap = folium.Map(location=[lat_c, lon_c], zoom_start=13, tiles=None, control_scale=True)
folium.TileLayer("Esri.WorldImagery", name="Esri basemap (today)").add_to(fmap)
folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(fmap)

for year, show in ((C.BEFORE_YEAR, False), (C.AFTER_YEAR, True)):
    url, bounds = true_colour(year)
    folium.raster_layers.ImageOverlay(url, bounds, name=f"Sentinel-2 {year} (true colour)",
                                      show=show).add_to(fmap)
for year, colour in ((C.BEFORE_YEAR, (34, 139, 34, 110)), (C.AFTER_YEAR, (0, 200, 120, 110))):
    url, bounds = mask_layer(C.OUT / f"forest_prob_{year}.tif", colour)
    folium.raster_layers.ImageOverlay(url, bounds, name=f"U-Net forest mask {year}",
                                      show=False).add_to(fmap)

cleared_geo = json.load(open(C.OUT / "cleared_areas.geojson"))
if cleared_geo["features"]:
    folium.GeoJson(
        cleared_geo,
        name=f"Cleared {C.BEFORE_YEAR}→{C.AFTER_YEAR}",
        style_function=lambda f: {"color": "#ff2d2d", "weight": 1.5, "fillColor": "#ff2d2d", "fillOpacity": 0.45},
        tooltip=folium.GeoJsonTooltip(["area_ha"], aliases=["Cleared area (ha):"]),
    ).add_to(fmap)
else:
    print("No cleared patches found — try loosening thresholds in config.py")
folium.Rectangle([[C.BBOX[1], C.BBOX[0]], [C.BBOX[3], C.BBOX[2]]], color="white",
                 weight=1, fill=False, dash_array="6", name="Study area").add_to(fmap)

panel = f"""
<div style="position:fixed;top:12px;left:56px;z-index:9999;background:rgba(20,20,20,.88);color:#eee;
padding:12px 14px;border-radius:8px;font:13px/1.45 system-ui,sans-serif;max-width:290px">
<b style="font-size:15px">Deforestation monitor — Rondônia, Brazil</b><br>
Sentinel-2 · U-Net segmentation · {C.BEFORE_YEAR} vs {C.AFTER_YEAR}<br><br>
<b style="color:#ff6b6b">{stats['cleared_ha']:,} ha</b> of forest cleared
({stats['cleared_pct_of_forest']}% of {C.BEFORE_YEAR} forest)<br>
{stats['n_patches']} patches · largest {stats['largest_patch_ha']} ha<br>
Model val IoU {metrics['best']['iou']:.3f} (NDVI baseline {metrics['ndvi_baseline']['iou']:.3f})<br>
<span style="color:#aaa">Toggle layers top-right · hover red patches</span></div>"""
fmap.get_root().html.add_child(folium.Element(panel))
Fullscreen().add_to(fmap); MiniMap(toggle_display=True).add_to(fmap)
folium.LayerControl(collapsed=False).add_to(fmap)
fmap.save(C.OUT / "deforestation_map.html")
print("Saved", C.OUT / "deforestation_map.html")
