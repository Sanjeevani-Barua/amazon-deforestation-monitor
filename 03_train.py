"""Step 3 — train the U-Net forest segmentation model (CPU-friendly).

Loss = BCE + Dice. Augmentation = random flips and 90-degree rotations.
Best model (by validation IoU) is saved to outputs/unet.pt.
"""
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import config as C
from common import UNet

torch.manual_seed(0)
np.random.seed(0)
torch.set_num_threads(max(1, torch.get_num_threads()))

norm = np.load(C.DATA / "norm.npz")
mean, std = norm["mean"], norm["std"]


def load(name):
    d = np.load(C.DATA / f"chips_{name}.npz")
    X = (d["X"] - mean[None, :, None, None]) / std[None, :, None, None]
    return torch.tensor(X, dtype=torch.float32), torch.tensor(d["Y"])[:, None], torch.tensor(d["M"])[:, None]


Xtr, Ytr, Mtr = load("train")
Xva, Yva, Mva = load("val")
print(f"train {len(Xtr)} chips | val {len(Xva)} chips")


def augment(x, y, m):
    k = np.random.randint(4)
    x, y, m = (torch.rot90(t, k, (2, 3)) for t in (x, y, m))
    if np.random.rand() < 0.5:
        x, y, m = (torch.flip(t, (3,)) for t in (x, y, m))
    return x, y, m


def loss_fn(logits, y, m):
    m = m.float()
    bce = (F.binary_cross_entropy_with_logits(logits, y, reduction="none") * m).sum() / m.sum()
    p = torch.sigmoid(logits) * m
    dice = 1 - (2 * (p * y).sum() + 1) / (p.sum() + (y * m).sum() + 1)
    return bce + dice


@torch.no_grad()
def evaluate(model):
    model.eval()
    tp = fp = fn = tn = 0
    for i in range(0, len(Xva), 32):
        pred = torch.sigmoid(model(Xva[i:i + 32])) > 0.5
        y, m = Yva[i:i + 32].bool(), Mva[i:i + 32]
        tp += (pred & y & m).sum().item(); fp += (pred & ~y & m).sum().item()
        fn += (~pred & y & m).sum().item(); tn += (~pred & ~y & m).sum().item()
    return {
        "iou": tp / max(tp + fp + fn, 1),
        "f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "accuracy": (tp + tn) / max(tp + tn + fp + fn, 1),
    }


model = UNet(in_ch=Xtr.shape[1], base=C.BASE_CHANNELS)
print(f"U-Net parameters: {sum(p.numel() for p in model.parameters()):,}")
opt = torch.optim.AdamW(model.parameters(), lr=C.LR, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, C.EPOCHS)

history, best = [], -1
for epoch in range(1, C.EPOCHS + 1):
    t0 = time.time()
    model.train()
    perm = torch.randperm(len(Xtr))
    total = 0.0
    for i in range(0, len(Xtr), C.BATCH_SIZE):
        idx = perm[i:i + C.BATCH_SIZE]
        x, y, m = augment(Xtr[idx], Ytr[idx], Mtr[idx])
        loss = loss_fn(model(x), y, m)
        opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item() * len(idx)
    sched.step()
    metrics = evaluate(model)
    metrics.update(epoch=epoch, train_loss=total / len(Xtr))
    history.append(metrics)
    flag = ""
    if metrics["iou"] > best:
        best = metrics["iou"]
        torch.save(model.state_dict(), C.OUT / "unet.pt")
        flag = "  <- saved"
    print(f"epoch {epoch:2d} | loss {metrics['train_loss']:.3f} | val IoU {metrics['iou']:.3f} "
          f"| F1 {metrics['f1']:.3f} | {time.time() - t0:.0f}s{flag}")

best_m = max(history, key=lambda h: h["iou"])
print("Best validation (U-Net):", {k: round(v, 3) for k, v in best_m.items()})

# Baseline: a simple NDVI threshold (threshold tuned on the TRAIN chips) — shows what the U-Net adds.
ndvi_tr = (Xtr[:, 6] * std[6] + mean[6]); ndvi_va = (Xva[:, 6] * std[6] + mean[6])
def iou_at(nd, y, m, t):
    p = nd > t; y = y.bool(); m = m.bool()
    tp = (p & y & m).sum().item(); fp = (p & ~y & m).sum().item(); fn = (~p & y & m).sum().item()
    return tp / max(tp + fp + fn, 1)
ts = np.arange(0.3, 0.95, 0.01)
t_best = ts[np.argmax([iou_at(ndvi_tr, Ytr[:, 0], Mtr[:, 0], t) for t in ts])]
baseline_iou = iou_at(ndvi_va, Yva[:, 0], Mva[:, 0], t_best)
print(f"Baseline NDVI > {t_best:.2f}: val IoU {baseline_iou:.3f}")
json.dump({"best": best_m, "ndvi_baseline": {"threshold": float(t_best), "iou": baseline_iou},
           "history": history}, open(C.OUT / "training_metrics.json", "w"), indent=2)

fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
ep = [h["epoch"] for h in history]
ax[0].plot(ep, [h["train_loss"] for h in history]); ax[0].set_title("Training loss (BCE + Dice)")
ax[1].plot(ep, [h["iou"] for h in history], label="IoU")
ax[1].plot(ep, [h["f1"] for h in history], label="F1"); ax[1].legend(); ax[1].set_title("Validation (spatial hold-out)")
for a in ax: a.set_xlabel("epoch")
fig.tight_layout(); fig.savefig(C.OUT / "training_curves.png", dpi=150)
