"""Step 2 — cut the training-year image + labels into 128x128 chips.

Train/validation split is SPATIAL: the right-hand strip of the area is held out.
A random split would put neighbouring (almost identical) chips in both sets and
inflate the validation score — the same leakage problem as splitting ECG data by beat.
"""
import numpy as np

import config as C
from raster_utils import add_indices, read_tif, valid_mask

img, _ = read_tif(C.composite_path(C.TRAIN_YEAR))
lab, _ = read_tif(C.LABEL_PATH)
feats = add_indices(img)
lab = lab[0]
valid = valid_mask(img)

H, W = lab.shape
split_col = int(W * (1 - C.VAL_FRACTION))
sets = {"train": ([], [], []), "val": ([], [], [])}
for y in range(0, H - C.CHIP + 1, C.CHIP):
    for x in range(0, W - C.CHIP + 1, C.CHIP):
        if x < split_col < x + C.CHIP:
            continue  # chip straddles the boundary -> drop it (buffer between sets)
        v = valid[y:y + C.CHIP, x:x + C.CHIP]
        if v.mean() < 0.9:
            continue  # too much cloud/no-data
        name = "train" if x + C.CHIP <= split_col else "val"
        X, Y, M = sets[name]
        X.append(feats[:, y:y + C.CHIP, x:x + C.CHIP])
        Y.append(lab[y:y + C.CHIP, x:x + C.CHIP])
        M.append(v)

for name, (X, Y, M) in sets.items():
    X, Y, M = np.stack(X), np.stack(Y).astype("float32"), np.stack(M)
    np.savez_compressed(C.DATA / f"chips_{name}.npz", X=X, Y=Y, M=M)
    print(f"{name}: {len(X)} chips, forest fraction {Y[M].mean():.1%}")

# Per-channel normalisation stats from TRAINING chips only
Xtr = np.load(C.DATA / "chips_train.npz")["X"]
mean = Xtr.mean(axis=(0, 2, 3))
std = Xtr.std(axis=(0, 2, 3)) + 1e-6
np.savez(C.DATA / "norm.npz", mean=mean, std=std)
print("Saved normalisation stats")
