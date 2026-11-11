"""Project settings. Change the area or years here — every script reads from this file."""
from pathlib import Path

# Area of interest: ~20 x 20 km near Jaci-Paraná, Rondônia (Brazilian Amazon),
# one of the most actively cleared areas in Brazil. (lon_min, lat_min, lon_max, lat_max)
BBOX = (-64.55, -9.75, -64.37, -9.57)
CRS = "EPSG:32720"          # UTM zone 20S (metres) — correct zone for this longitude
RESOLUTION = 10             # metres per pixel (Sentinel-2 native for RGB/NIR)

# Years: train on 2021 (matches ESA WorldCover 2021 labels), then compare 2019 vs 2024.
TRAIN_YEAR = 2021
BEFORE_YEAR = 2019
AFTER_YEAR = 2024
SEASON = ("06-01", "09-30")  # Amazon dry season = fewest clouds
MAX_CLOUD = 40               # % scene cloud cover allowed when searching
MAX_SCENES = 12              # scenes per composite (more = cleaner, slower download)

# Sentinel-2 bands used: Blue, Green, Red, NIR, SWIR1, SWIR2
BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]

# Chips + training (sized to run on a laptop CPU)
CHIP = 128
VAL_FRACTION = 0.25          # right-hand 25% of the area is held out for validation
EPOCHS = 25
BATCH_SIZE = 16
LR = 1e-3
BASE_CHANNELS = 16           # U-Net width; 16 keeps it ~0.5M params

# Change detection
P_FOREST_BEFORE = 0.6        # must be confidently forest before ...
P_FOREST_AFTER = 0.4         # ... and confidently not-forest after
NDVI_DROP = 0.15             # plus a real drop in vegetation index
MIN_AREA_HA = 0.5            # ignore patches smaller than this

DATA = Path(__file__).parent / "data"
OUT = Path(__file__).parent / "outputs"
DATA.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)


def composite_path(year):
    return DATA / f"s2_{year}.tif"


LABEL_PATH = DATA / f"worldcover_{TRAIN_YEAR}.tif"
