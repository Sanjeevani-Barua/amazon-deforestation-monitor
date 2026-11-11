"""Raster helpers that do NOT need PyTorch (used by 02_make_chips.py)."""
import numpy as np
import rasterio


def read_tif(path):
    with rasterio.open(path) as src:
        return src.read(), src.profile


def add_indices(img):
    """img: (6, H, W) reflectance [B,G,R,NIR,SWIR1,SWIR2] -> (8, H, W) with NDVI and NBR.
    NDVI responds to green vegetation; NBR (NIR vs SWIR2) is very sensitive to clearing/burning."""
    nir, red, swir2 = img[3], img[2], img[5]
    ndvi = (nir - red) / (nir + red + 1e-6)
    nbr = (nir - swir2) / (nir + swir2 + 1e-6)
    return np.concatenate([img, ndvi[None], nbr[None]]).astype("float32")


def valid_mask(img):
    return (img[:6] > 0).all(0)
