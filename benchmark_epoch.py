"""
Benchmark a single Cellpose training epoch to find the bottleneck.
Loads 64 tiles from the cache and times each stage.
"""
import time
import numpy as np
import tifffile
import torch
from pathlib import Path
from cellpose import models, train, dynamics
from cellpose.transforms import random_rotate_and_resize, normalize_img

cache_dir = Path("annotations/training_data/parasites/_cellpose_tiles")
tile_files = sorted(cache_dir.glob("*_flows.tif"))[:64]
img_files = [p.parent / p.name.replace("_flows.tif", ".tif") for p in tile_files]

print(f"Loading {len(tile_files)} tiles...", flush=True)
t0 = time.time()
train_imgs = [tifffile.imread(str(f)).astype(np.float32) for f in img_files]
train_flows = [tifffile.imread(str(f)).astype(np.float32) for f in tile_files]
print(f"  Loaded in {time.time()-t0:.2f}s", flush=True)
print(f"  img shape: {train_imgs[0].shape}, flow shape: {train_flows[0].shape}", flush=True)

use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")
print(f"  GPU: {use_gpu}", flush=True)

print("Loading model...", flush=True)
t0 = time.time()
cp_model = models.CellposeModel(gpu=use_gpu, model_type="cyto3")
print(f"  Model loaded in {time.time()-t0:.2f}s", flush=True)

print("Running labels_to_flows...", flush=True)
t0 = time.time()
flows_out = dynamics.labels_to_flows(train_flows, files=None, device=device)
print(f"  labels_to_flows done in {time.time()-t0:.2f}s", flush=True)

print("Running _reshape_norm...", flush=True)
t0 = time.time()
imgs_normed = train._reshape_norm(train_imgs)
print(f"  _reshape_norm done in {time.time()-t0:.2f}s", flush=True)

print("Timing augmentation (10 batches of 1)...", flush=True)
# Keep bfloat16 — do NOT convert to float32
optimizer = torch.optim.AdamW(cp_model.net.parameters(), lr=0.005, weight_decay=1e-5)
cp_model.net.train()

aug_times, fwd_times = [], []
for trial in range(10):
    idx = trial % len(imgs_normed)
    imgs_batch = [imgs_normed[idx]]
    lbls_batch = [flows_out[idx][1:]]
    rsc = np.ones(1, "float32")

    t0 = time.time()
    imgi, lbl = random_rotate_and_resize(imgs_batch, Y=lbls_batch, rescale=rsc, xy=(256, 256))[:2]
    t1 = time.time()
    X = torch.from_numpy(imgi).to(device, dtype=cp_model.net.dtype)
    lbl_t = torch.from_numpy(lbl).to(device)
    with torch.autocast(device_type=device.type, dtype=cp_model.net.dtype):
        y = cp_model.net(X)[0]
    loss = train._loss_fn_seg(lbl_t, y, device)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    t2 = time.time()
    aug_times.append(t1 - t0)
    fwd_times.append(t2 - t1)

print(f"  Augmentation per batch: {np.mean(aug_times):.3f}s (mean of 10)", flush=True)
print(f"  Forward+backward per batch: {np.mean(fwd_times):.3f}s (mean of 10)", flush=True)
total = np.mean(aug_times) + np.mean(fwd_times)
print(f"  Total per batch: {total:.3f}s", flush=True)
print(f"  Projected 64 batches/epoch: {total*64:.1f}s", flush=True)
print(f"  Projected 100 epochs: {total*64*100/60:.1f} minutes", flush=True)
