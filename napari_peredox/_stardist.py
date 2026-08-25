"""
_stardist.py — Model fine-tuning and inference for PV segmentation

Two models are trained and maintained in parallel:

  ``vacuoles``  — StarDist2D fine-tuned from '2D_versatile_fluo'.
                  Detects whole parasitophorous vacuoles as star-convex polygons.
                  Used in Stage 1 of the two-stage pipeline.
                  NOTE: StarDist uses TensorFlow which has no GPU support on
                  native Windows >= TF 2.11.  Runs on CPU (use WSL2 for GPU).

  ``parasites`` — Cellpose fine-tuned from 'cyto3', running on PyTorch + CUDA.
                  Detects individual parasite bodies within vacuoles.
                  Used in Stage 2 (run per-vacuole crop).
                  Full GPU support on Windows via PyTorch/CUDA.

Model save locations:
  annotations_dir/stardist_model/vacuoles/peredox_vacuoles/   (StarDist)
  annotations_dir/stardist_model/parasites/peredox_parasites.pth  (Cellpose)

Training data lives in:
  annotations_dir/training_data/vacuoles/images/  + masks/
  annotations_dir/training_data/parasites/images/ + masks/
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Literal

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ModelMode = Literal["vacuoles", "parasites"]

_MODEL_NAMES: dict[ModelMode, str] = {
    "vacuoles": "peredox_vacuoles",
    "parasites": "peredox_parasites",
}

_TRAINING_SUBDIRS: dict[ModelMode, str] = {
    "vacuoles": "training_data/vacuoles",
    "parasites": "training_data/parasites",
}

# Legacy constant kept for backward compatibility with existing code that
# imports TRAINING_SUBDIR / MODEL_NAME directly.
TRAINING_SUBDIR = "training_data/vacuoles"
MODEL_NAME = "peredox_vacuoles"


# ---------------------------------------------------------------------------
# Data discovery
# ---------------------------------------------------------------------------


def discover_training_pairs(
    training_dir: Path,
    mode: ModelMode = "vacuoles",
) -> list[tuple[Path, Path]]:
    """
    Return sorted (image_path, mask_path) pairs for *mode* in training_dir.

    Directory structure expected:
      training_dir/<mode>/images/<stem>.tif
      training_dir/<mode>/masks/<stem>_mask.tif
    """
    base = Path(training_dir) / mode
    img_dir = base / "images"
    mask_dir = base / "masks"
    if not img_dir.exists() or not mask_dir.exists():
        return []
    pairs = []
    for img_path in sorted(img_dir.glob("*.tif")):
        # For vacuoles: prefer the fuller _vac_mask.tif, fall back to _mask.tif
        if mode == "vacuoles":
            vac_mask = mask_dir / f"{img_path.stem}_vac_mask.tif"
            plain_mask = mask_dir / f"{img_path.stem}_mask.tif"
            mask_path = vac_mask if vac_mask.exists() else plain_mask
        else:
            mask_path = mask_dir / f"{img_path.stem}_mask.tif"
        if mask_path.exists():
            pairs.append((img_path, mask_path))
    return pairs


def count_training_pairs(
    training_dir: Path,
    mode: ModelMode = "vacuoles",
) -> int:
    """Return the number of complete training pairs for *mode*."""
    return len(discover_training_pairs(Path(training_dir), mode))


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def _normalise(img: np.ndarray, lo: float = 1.0, hi: float = 99.8) -> np.ndarray:
    """Percentile-normalise a 2-D float image to [0, 1]."""
    lo_val = float(np.percentile(img, lo))
    hi_val = float(np.percentile(img, hi))
    img = (img.astype(np.float32) - lo_val) / max(hi_val - lo_val, 1e-6)
    return np.clip(img, 0.0, 1.0)


def _load_pair(
    img_path: Path,
    mask_path: Path,
    seg_channel: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Load and normalise one (image, mask) pair.

    The saved image TIFF is (C, H, W); we extract one channel and
    percentile-normalise it to [0, 1].  Returns None on any read error.
    """
    import tifffile

    try:
        raw = tifffile.imread(str(img_path)).astype(np.float32)
        mask = tifffile.imread(str(mask_path)).astype(np.int32)
    except Exception:
        return None

    if raw.ndim == 2:
        img = raw
    elif raw.ndim == 3:
        ch = min(seg_channel, raw.shape[0] - 1)
        img = raw[ch]
    else:
        return None

    if img.shape != mask.shape:
        return None

    return _normalise(img), mask


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------


def build_dataset(
    training_dir: Path,
    mode: ModelMode = "vacuoles",
    seg_channel: int = 0,
    val_fraction: float = 0.15,
    rng_seed: int = 42,
) -> tuple[list, list, list, list]:
    """
    Load all training pairs for *mode* and split into train / validation sets.

    Returns
    -------
    (X_train, Y_train, X_val, Y_val)
    """
    pairs = discover_training_pairs(training_dir, mode)
    if not pairs:
        raise ValueError(f"No training pairs found for mode '{mode}' in {training_dir}")

    rng = np.random.default_rng(rng_seed)
    indices = rng.permutation(len(pairs)).tolist()
    n_val = max(1, int(len(pairs) * val_fraction))
    val_set = set(indices[:n_val])

    X_train, Y_train, X_val, Y_val = [], [], [], []
    for i, (img_path, mask_path) in enumerate(pairs):
        result = _load_pair(img_path, mask_path, seg_channel)
        if result is None:
            continue
        img, mask = result
        if i in val_set:
            X_val.append(img)
            Y_val.append(mask)
        else:
            X_train.append(img)
            Y_train.append(mask)

    if not X_train:
        raise ValueError("All training images failed to load — check TIFF files.")

    return X_train, Y_train, X_val, Y_val


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------


def _augmenter(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Random 90° rotations and axis flips applied consistently to image and mask."""
    k = np.random.randint(4)
    x, y = np.rot90(x, k), np.rot90(y, k)
    if np.random.rand() > 0.5:
        x, y = np.fliplr(x), np.fliplr(y)
    if np.random.rand() > 0.5:
        x, y = np.flipud(x), np.flipud(y)
    return x, y


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_stardist(
    training_dir: Path,
    model_dir: Path,
    mode: ModelMode = "vacuoles",
    seg_channel: int = 0,
    n_epochs: int = 100,
    patch_size: tuple[int, int] = (256, 256),
    val_fraction: float = 0.15,
    progress_cb: Callable[[str], None] | None = None,
) -> Path:
    """
    Fine-tune a model for *mode*.

    - ``parasites``: Cellpose fine-tuning on PyTorch/CUDA (GPU on Windows).
    - ``vacuoles``:  StarDist2D fine-tuning from '2D_versatile_fluo' (CPU on
                     native Windows; use WSL2 for GPU).

    Returns the path to the saved model file/directory.
    """
    if mode == "parasites":
        return _train_cellpose_parasites(
            training_dir=Path(training_dir),
            model_dir=Path(model_dir),
            seg_channel=seg_channel,
            n_epochs=n_epochs,
            val_fraction=val_fraction,
            progress_cb=progress_cb,
        )
    return _train_stardist_vacuoles(
        training_dir=Path(training_dir),
        model_dir=Path(model_dir),
        seg_channel=seg_channel,
        n_epochs=n_epochs,
        patch_size=patch_size,
        val_fraction=val_fraction,
        progress_cb=progress_cb,
    )


def _train_cellpose_parasites(
    training_dir: Path,
    model_dir: Path,
    seg_channel: int = 0,
    n_epochs: int = 100,
    val_fraction: float = 0.15,
    progress_cb: Callable[[str], None] | None = None,
) -> Path:
    """
    Fine-tune Cellpose 'cyto3' on parasite training data using PyTorch/CUDA.

    Images are (C, H, W) float32 TIFFs; masks are (H, W) int32 TIFFs.
    The fine-tuned model is saved as:
      model_dir/parasites/peredox_parasites.pth
    """
    import tifffile
    import torch
    from cellpose import models, train

    def log(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)

    use_gpu = False  # force CPU — laptop GPU has insufficient free VRAM
    log(
        f"[parasites] GPU: {'yes — ' + torch.cuda.get_device_name(0) if use_gpu else 'no — running on CPU'}"
    )

    log("[parasites] Loading training data…")
    pairs = discover_training_pairs(training_dir, mode="parasites")
    if not pairs:
        raise ValueError(f"No parasite training pairs found in {training_dir}")

    rng = np.random.default_rng(42)
    indices = rng.permutation(len(pairs)).tolist()
    n_val = max(1, int(len(pairs) * val_fraction))
    val_idx_set = set(indices[:n_val])

    from cellpose import dynamics

    # Tile images into TILE_SZ×TILE_SZ patches so each batch read is small.
    # Full 2048×2048 flow TIFFs (~64 MB each) cause severe I/O bottleneck when
    # read per-batch with load_files=False.  512×512 patches are ~4 MB each and
    # load in microseconds.  Only non-background tiles (≥1 labelled pixel) are kept.
    TILE_SZ = 512

    cache_dir = training_dir / "parasites" / "_cellpose_tiles"
    cache_dir.mkdir(parents=True, exist_ok=True)

    all_img_files: list[str] = []
    all_flow_files: list[str] = []
    all_in_val: list[bool] = []

    log(f"[parasites] Tiling into {TILE_SZ}×{TILE_SZ} patches…")
    device = torch.device("cuda" if use_gpu else "cpu")
    n_total = len(pairs)
    n_tiles_new = 0

    for i, (img_path, mask_path) in enumerate(pairs):
        if (i + 1) % 20 == 0 or i == n_total - 1:
            log(
                f"[parasites]   {i + 1}/{n_total} images tiled ({n_tiles_new} new tiles)…"
            )
        try:
            raw = tifffile.imread(str(img_path)).astype(np.float32)
            mask = tifffile.imread(str(mask_path)).astype(np.int32)
        except Exception:
            continue
        img2d = raw[min(seg_channel, raw.shape[0] - 1)] if raw.ndim == 3 else raw
        H, W = img2d.shape

        for r in range(0, H, TILE_SZ):
            for c in range(0, W, TILE_SZ):
                tile_mask = mask[r : r + TILE_SZ, c : c + TILE_SZ]
                if tile_mask.max() == 0:
                    continue  # skip empty tiles
                stem = f"{img_path.stem}_r{r:04d}_c{c:04d}"
                out_img = cache_dir / f"{stem}.tif"
                flows_path = cache_dir / f"{stem}_flows.tif"

                if not out_img.exists():
                    tile_img = img2d[r : r + TILE_SZ, c : c + TILE_SZ]
                    tifffile.imwrite(str(out_img), tile_img)

                if not flows_path.exists():
                    flows = dynamics.labels_to_flows(
                        [tile_mask], files=None, device=device
                    )[0]
                    tifffile.imwrite(str(flows_path), flows)
                    n_tiles_new += 1

                all_img_files.append(str(out_img))
                all_flow_files.append(str(flows_path))
                all_in_val.append(i in val_idx_set)

    train_img_files = [f for f, v in zip(all_img_files, all_in_val) if not v]
    train_flow_files = [f for f, v in zip(all_flow_files, all_in_val) if not v]
    val_img_files = [f for f, v in zip(all_img_files, all_in_val) if v]
    val_flow_files = [f for f, v in zip(all_flow_files, all_in_val) if v]

    # Subsample tiles so the full dataset fits in RAM (~512 MB for 500 tiles).
    # Cellpose's load_files=False reads from disk every batch — catastrophically
    # slow for any tile count.  Loading arrays directly is the only fast path.
    MAX_TRAIN_TILES = 500
    MAX_VAL_TILES = 100
    rng2 = np.random.default_rng(0)
    if len(train_img_files) > MAX_TRAIN_TILES:
        keep = rng2.choice(len(train_img_files), MAX_TRAIN_TILES, replace=False)
        train_img_files = [train_img_files[k] for k in keep]
        train_flow_files = [train_flow_files[k] for k in keep]
    if len(val_img_files) > MAX_VAL_TILES:
        keep = rng2.choice(len(val_img_files), MAX_VAL_TILES, replace=False)
        val_img_files = [val_img_files[k] for k in keep]
        val_flow_files = [val_flow_files[k] for k in keep]

    log(
        f"[parasites] Loading {len(train_img_files)} train + {len(val_img_files)} val tiles into RAM…"
    )
    train_imgs = [tifffile.imread(f).astype(np.float32) for f in train_img_files]
    # Flow files are (4,H,W); labels_to_flows expects the full 4-channel array
    train_flows = [tifffile.imread(f).astype(np.float32) for f in train_flow_files]
    val_imgs = [tifffile.imread(f).astype(np.float32) for f in val_img_files]
    val_flows = [tifffile.imread(f).astype(np.float32) for f in val_flow_files]

    save_dir = model_dir / "parasites"
    save_dir.mkdir(parents=True, exist_ok=True)
    model_name = "peredox_parasites"

    log("[parasites] Loading Cellpose 'cyto3' pretrained weights…")
    cp_model = models.CellposeModel(gpu=use_gpu, model_type="cyto3")

    log(f"[parasites] Fine-tuning for {n_epochs} epochs…")
    # Pass data as arrays — avoids all per-batch disk I/O.
    # labels_to_flows sees shape (4,H,W) → "flows precomputed" branch → no recompute.
    model_path, train_losses, val_losses = train.train_seg(
        cp_model.net,
        train_data=train_imgs,
        train_labels=train_flows,
        test_data=val_imgs if val_imgs else None,
        test_labels=val_flows if val_imgs else None,
        min_train_masks=0,
        save_path=str(save_dir),
        save_every=n_epochs,
        n_epochs=n_epochs,
        learning_rate=0.005,
        weight_decay=1e-5,
        nimg_per_epoch=8,
        model_name=model_name,
    )

    out = Path(model_path)
    log(f"[parasites] Model saved -> {out}")
    log(
        f"[parasites] Final train loss: {train_losses[-1]:.4f}"
        + (f"  val loss: {val_losses[-1]:.4f}" if val_losses else "")
    )
    return out


def _train_stardist_vacuoles(
    training_dir: Path,
    model_dir: Path,
    seg_channel: int = 0,
    n_epochs: int = 100,
    patch_size: tuple[int, int] = (256, 256),
    val_fraction: float = 0.15,
    progress_cb: Callable[[str], None] | None = None,
) -> Path:
    """Fine-tune StarDist2D for vacuole detection (CPU on native Windows)."""
    from stardist.models import Config2D, StarDist2D

    model_name = _MODEL_NAMES["vacuoles"]

    def log(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)

    log("[vacuoles] Loading training data…")
    X_tr, Y_tr, X_val, Y_val = build_dataset(
        training_dir,
        mode="vacuoles",
        seg_channel=seg_channel,
        val_fraction=val_fraction,
    )
    log(f"[vacuoles] Dataset: {len(X_tr)} train  +  {len(X_val)} validation images.")

    log("[vacuoles] Loading '2D_versatile_fluo' pretrained weights…")
    pretrained = StarDist2D.from_pretrained("2D_versatile_fluo")

    out_dir = model_dir / "vacuoles"
    out_dir.mkdir(parents=True, exist_ok=True)

    conf = Config2D(
        n_rays=pretrained.config.n_rays,
        grid=pretrained.config.grid,
        n_channel_in=1,
        train_patch_size=patch_size,
        train_epochs=n_epochs,
        train_steps_per_epoch=max(100, min(400, len(X_tr) * 4)),
        train_batch_size=8,
    )
    model = StarDist2D(conf, name=model_name, basedir=str(out_dir))

    log("[vacuoles] Transferring pretrained weights…")
    try:
        model.keras_model.set_weights(pretrained.keras_model.get_weights())
        log("[vacuoles]   Weights transferred successfully.")
    except Exception as exc:
        log(
            f"[vacuoles]   Weight transfer skipped ({exc}) — training from random init."
        )

    log(f"[vacuoles] Fine-tuning for {n_epochs} epochs…")
    model.train(X_tr, Y_tr, validation_data=(X_val, Y_val), augmenter=_augmenter)

    log("[vacuoles] Optimising detection thresholds…")
    model.optimize_thresholds(X_val, Y_val)

    out_path = out_dir / model_name
    log(f"[vacuoles] Model saved -> {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Model loading / inference
# ---------------------------------------------------------------------------


def load_stardist_model(
    model_dir: Path,
    mode: ModelMode = "vacuoles",
):
    """
    Load a fine-tuned model for *mode*.

    - parasites: returns a CellposeModel loaded from the .pth file.
    - vacuoles:  returns a StarDist2D model.

    Returns None if the model file/directory does not exist.
    """
    model_dir = Path(model_dir)

    if mode == "parasites":
        para_dir = model_dir / "parasites"
        # Cellpose saves as <name>_<timestamp> with no extension, or .pth
        candidates = (
            list(para_dir.glob("peredox_parasites*")) if para_dir.exists() else []
        )
        if not candidates:
            return None
        # Pick most recently modified
        model_path = max(candidates, key=lambda p: p.stat().st_mtime)
        try:
            import torch
            from cellpose import models

            use_gpu = torch.cuda.is_available()
            cp_model = models.CellposeModel(
                gpu=use_gpu, pretrained_model=str(model_path)
            )
            # Tag so predict_stardist knows which backend to use
            cp_model._peredox_backend = "cellpose"
            return cp_model
        except Exception:
            return None

    # vacuoles — StarDist2D
    from stardist.models import StarDist2D

    model_name = _MODEL_NAMES[mode]
    base = model_dir / mode
    model_path = base / model_name
    if not model_path.exists():
        return None
    try:
        m = StarDist2D(None, name=model_name, basedir=str(base))
        m._peredox_backend = "stardist"
        return m
    except Exception:
        return None


def predict_stardist(
    image: np.ndarray,
    model,
    seg_channel: int = 0,
) -> np.ndarray:
    """
    Run inference with either a StarDist or Cellpose model.

    Parameters
    ----------
    image : np.ndarray
        (H, W) or (H, W, C) float image.
    model : StarDist2D | CellposeModel
        Loaded model (tagged with ._peredox_backend).
    seg_channel : int
        Channel to use if image is multi-channel.

    Returns
    -------
    np.ndarray (H, W) int32 instance label array.
    """
    backend = getattr(model, "_peredox_backend", "stardist")

    if image.ndim == 2:
        img2d = image.astype(np.float32)
    else:
        ch = min(seg_channel, image.shape[-1] - 1)
        img2d = image[..., ch].astype(np.float32)

    if backend == "cellpose":
        masks, _, _ = model.eval(img2d, diameter=None, channels=[0, 0])
        return np.asarray(masks).astype(np.int32)

    # StarDist
    img2d = _normalise(img2d)
    labels, _ = model.predict_instances(img2d)
    return labels.astype(np.int32)
