"""Shared helpers: the U-Net and tiled prediction (needs PyTorch)."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from raster_utils import add_indices, read_tif, valid_mask  # noqa: F401 (re-exported)


# ---------------------------------------------------------------- U-Net
class Block(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """Small 4-level U-Net. Input must have H, W divisible by 8."""

    def __init__(self, in_ch=8, base=16):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8]
        self.enc = nn.ModuleList([Block(in_ch, c[0]), Block(c[0], c[1]), Block(c[1], c[2])])
        self.bottleneck = Block(c[2], c[3])
        self.up = nn.ModuleList([nn.ConvTranspose2d(c[i + 1], c[i], 2, stride=2) for i in (2, 1, 0)])
        self.dec = nn.ModuleList([Block(c[i] * 2, c[i]) for i in (2, 1, 0)])
        self.head = nn.Conv2d(c[0], 1, 1)
        self.drop = nn.Dropout2d(0.1)

    def forward(self, x):
        skips = []
        for enc in self.enc:
            x = enc(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.drop(self.bottleneck(x))
        for up, dec, skip in zip(self.up, self.dec, reversed(skips)):
            x = dec(torch.cat([up(x), skip], 1))
        return self.head(x)  # logits


# ---------------------------------------------------------------- inference
@torch.no_grad()
def predict_image(model, feats, mean, std, tile=512, overlap=64):
    """Sliding-window prediction over a full (C, H, W) image -> forest probability (H, W)."""
    model.eval()
    x = (feats - mean[:, None, None]) / std[:, None, None]
    C, H, W = x.shape
    out = np.zeros((H, W), "float32")
    weight = np.zeros((H, W), "float32")
    step = tile - overlap
    for y0 in range(0, max(H - overlap, 1), step):
        for x0 in range(0, max(W - overlap, 1), step):
            y1, x1 = min(y0 + tile, H), min(x0 + tile, W)
            patch = x[:, y0:y1, x0:x1]
            ph, pw = (-patch.shape[1]) % 8, (-patch.shape[2]) % 8   # pad to multiple of 8
            patch = np.pad(patch, ((0, 0), (0, ph), (0, pw)), mode="reflect")
            prob = torch.sigmoid(model(torch.from_numpy(patch)[None]))[0, 0].numpy()
            out[y0:y1, x0:x1] += prob[: y1 - y0, : x1 - x0]
            weight[y0:y1, x0:x1] += 1
    return out / np.maximum(weight, 1)
