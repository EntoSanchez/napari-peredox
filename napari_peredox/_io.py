"""
_io.py — Persistence layer for napari-peredox

Purpose
-------
Handles all reading and writing of files produced by the plugin:

  annotations/
  ├── curated_features.csv      — growing log of annotated PVs (never overwritten,
  │                               only appended; re-curating the same image+label
  │                               overwrites just that row)
  ├── curated_features.joblib   — trained RandomForest classifier (overwritten
  │                               each time 'Save & retrain' is clicked)
  └── results/
      ├── <stem>_measurements.csv — per-image measurement table (one row per PV)
      └── <stem>_labels.tif       — int32 label image for the same image

The `image_stem` (e.g. "experiment_01_pos003") is used to tie together the
CSV, TIFF, and annotation rows from a single image.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ── Saving segmentation outputs ───────────────────────────────────────────────


def save_measurements(
    df: pd.DataFrame,
    image_stem: str,
    annotations_dir: str | Path,
) -> Path:
    """
    Write the per-PV measurements DataFrame to a CSV file.

    The file is placed in annotations_dir/results/<image_stem>_measurements.csv.
    If the results/ subdirectory does not exist it is created automatically.

    Parameters
    ----------
    df : pd.DataFrame
        Output of measure_pvs(), indexed by label id.
    image_stem : str
        Short name for the source image (used as filename prefix).
    annotations_dir : str or Path
        Root annotations directory (default: <project>/annotations/).

    Returns
    -------
    out_path : Path
        Full path of the saved CSV file.
    """
    out_dir = Path(annotations_dir) / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{image_stem}_measurements.csv"
    df.to_csv(out_path)
    return out_path


def save_labels(
    labels: np.ndarray,
    image_stem: str,
    annotations_dir: str | Path,
) -> Path:
    """
    Save the integer label image as a TIFF so it can be reloaded into napari
    or inspected in Fiji.

    Parameters
    ----------
    labels : np.ndarray (H, W) int32
        Label image from segment_pvs().
    image_stem : str
        Short name for the source image (used as filename prefix).
    annotations_dir : str or Path
        Root annotations directory.

    Returns
    -------
    out_path : Path
        Full path of the saved TIFF.
    """
    import tifffile

    out_dir = Path(annotations_dir) / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{image_stem}_labels.tif"

    # Ensure int32 to avoid potential TIFF dtype issues with large label values
    tifffile.imwrite(str(out_path), labels.astype(np.int32))
    return out_path


# ── Curation log management ───────────────────────────────────────────────────


def append_curated_annotations(
    decisions: dict[int, int],
    features: pd.DataFrame,
    image_stem: str,
    annotations_dir: str | Path,
    vacuole_assignments: dict[int, int] | None = None,
    csv_name: str = "curated_features.csv",
) -> Path:
    """
    Append accept/reject decisions (with features) to the running annotation CSV.

    This is the function that grows the training dataset over time.  It is
    designed to be safe to call repeatedly:
      - Skipped segments (decision == -1) are excluded.
      - If the same image+label pair was annotated in a previous session, the
        old row is replaced so each PV only appears once in the training set.

    Parameters
    ----------
    decisions : dict {label_id → int}
        Values: 1 = accept (true PV), 0 = reject (false positive), -1 = skip.
    features : pd.DataFrame
        Feature table from extract_features(), indexed by label_id.
        If a label id is missing from features, only the image_stem/label/accepted
        columns are saved (the row will be unusable for training but won't crash).
    image_stem : str
        Source image name, used to group rows in the CSV.
    annotations_dir : str or Path
        Root annotations directory where curated_features.csv lives.
    vacuole_assignments : dict {label_id → vacuole_id}, optional
        Vacuole group assignments from the curation gallery.  Saved as a
        `vacuole_id` column so `learn_dilation_radius()` can calibrate the
        grouping radius from ground truth parasite pairings.

    Returns
    -------
    csv_path : Path
        Path to the (updated) curated_features.csv.
    """
    annotations_dir = Path(annotations_dir)
    annotations_dir.mkdir(parents=True, exist_ok=True)
    csv_path = annotations_dir / csv_name

    # ── Step 1: build the new rows ────────────────────────────────────────────
    # Only include rows where the user made an actual decision (1 or 0)
    decided = {lid: dec for lid, dec in decisions.items() if dec in (0, 1)}
    if not decided:
        return csv_path  # Nothing to save

    rows = []
    for lid, dec in decided.items():
        row = {
            "image_stem": image_stem,
            "label": lid,
            "accepted": dec,
        }
        # Attach the vacuole assignment if provided
        if vacuole_assignments is not None and lid in vacuole_assignments:
            row["vacuole_id"] = vacuole_assignments[lid]
        # Attach the feature values if available for this label
        if isinstance(features, pd.DataFrame) and lid in features.index:
            row.update(features.loc[lid].to_dict())
        rows.append(row)

    new_df = pd.DataFrame(rows)

    # ── Step 2: merge with existing annotations ────────────────────────────────
    if csv_path.exists():
        existing = pd.read_csv(csv_path)

        # Remove any rows that are being overwritten (same image_stem + label).
        # Build a set of (image_stem, label) pairs that are being updated
        update_index = set(zip(new_df["image_stem"], new_df["label"]))
        # Keep only existing rows that are NOT in the update set
        mask_keep = ~existing.apply(
            lambda r: (r["image_stem"], r["label"]) in update_index, axis=1
        )
        existing = existing[mask_keep]

        # Concatenate old (minus overwritten) + new
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        # First time saving — the new rows are the entire file
        combined = new_df

    combined.to_csv(csv_path, index=False)
    return csv_path


# ── Loading previously saved data ─────────────────────────────────────────────


def load_measurements(
    image_stem: str,
    annotations_dir: str | Path,
) -> pd.DataFrame | None:
    """
    Load a previously saved measurement CSV for an image.

    Returns None if no file is found (i.e. this image has not been processed
    in a previous session).

    Parameters
    ----------
    image_stem : str
        Same stem used when the file was saved.
    annotations_dir : str or Path
        Root annotations directory.
    """
    path = Path(annotations_dir) / "results" / f"{image_stem}_measurements.csv"
    if path.exists():
        return pd.read_csv(path, index_col="label")
    return None


def load_labels(
    image_stem: str,
    annotations_dir: str | Path,
) -> np.ndarray | None:
    """
    Load a previously saved label TIFF for an image.

    Returns None if the file does not exist.
    """
    import tifffile

    path = Path(annotations_dir) / "results" / f"{image_stem}_labels.tif"
    if path.exists():
        return tifffile.imread(str(path)).astype(np.int32)
    return None


# ── StarDist training data collection ─────────────────────────────────────────


def _build_vacuole_mask(
    labels: np.ndarray,
    vacuole_map: dict[int, int],
    decisions: dict[int, int],
) -> np.ndarray:
    """
    Build a per-vacuole label image for the vacuole StarDist model.

    For each vacuole, the union of accepted parasite pixels is dilated (4 px)
    to bridge internal gaps and hole-filled to produce a compact single region.
    """
    from collections import defaultdict

    from scipy.ndimage import binary_fill_holes
    from skimage.morphology import dilation, disk

    rejected = {lid for lid, dec in decisions.items() if dec == 0}

    vac_parasites: dict[int, list[int]] = defaultdict(list)
    for para_label, vac_id in vacuole_map.items():
        if para_label not in rejected:
            vac_parasites[vac_id].append(para_label)

    vac_mask = np.zeros_like(labels, dtype=np.int32)
    selem = disk(4)
    for vac_id, para_labels in vac_parasites.items():
        union = np.zeros(labels.shape, dtype=bool)
        for pl in para_labels:
            union |= labels == pl
        if not union.any():
            continue
        dilated = dilation(union, selem)
        filled = binary_fill_holes(dilated)
        vac_mask[filled] = vac_id

    return vac_mask


def _build_parasite_mask(
    labels: np.ndarray,
    decisions: dict[int, int],
) -> np.ndarray:
    """
    Build a per-parasite label image for the parasite StarDist model.

    Keeps all individual parasite labels except explicitly rejected ones.
    """
    rejected = {lid for lid, dec in decisions.items() if dec == 0}
    if not rejected:
        return labels.copy()
    curated = labels.copy()
    for lid in rejected:
        curated[labels == lid] = 0
    return curated


def _write_tiff_pair(
    image: np.ndarray,
    mask: np.ndarray,
    stem: str,
    base_dir: Path,
) -> tuple[Path, Path]:
    """Write (C,H,W) image + int32 mask TIFFs into base_dir/images/ and masks/."""
    import tifffile

    img_dir = base_dir / "images"
    mask_dir = base_dir / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    if image.ndim == 2:
        img_chw = image[np.newaxis].astype(np.float32)
    else:
        img_chw = np.moveaxis(image, -1, 0).astype(np.float32)

    img_path = img_dir / f"{stem}.tif"
    mask_path = mask_dir / f"{stem}_mask.tif"
    tifffile.imwrite(str(img_path), img_chw, imagej=True)
    tifffile.imwrite(str(mask_path), mask.astype(np.int32))
    return img_path, mask_path


def save_training_pair(
    image: np.ndarray,
    labels: np.ndarray,
    decisions: dict[int, int],
    stem: str,
    training_dir: str | Path,
    vacuole_map: dict[int, int] | None = None,
) -> dict[str, tuple[Path, Path]]:
    """
    Save (image, mask) TIFF pairs for both StarDist models simultaneously.

    Two pairs are written per curation session:

    * ``vacuoles`` — one filled region per PV (requires vacuole_map).
      If vacuole_map is None only the parasite pair is written.
    * ``parasites`` — one label per individual parasite body (rejected labels
      zeroed out).

    Parameters
    ----------
    image : np.ndarray (H, W, C) float32
    labels : np.ndarray (H, W) int32
        Per-parasite label array from segmentation / curation.
    decisions : dict {label_id → 0/1/-1}
    stem : str
        Unique name for this image.
    training_dir : str or Path
        Root training-data folder.  Sub-folders vacuoles/ and parasites/ are
        created automatically inside it.
    vacuole_map : dict {parasite_label → vacuole_id}, optional
        Required for the vacuole mask.  When absent only the parasite pair
        is saved.

    Returns
    -------
    dict with keys 'vacuoles' and/or 'parasites', each mapping to
    (img_path, mask_path).
    """
    training_dir = Path(training_dir)
    result: dict[str, tuple[Path, Path]] = {}

    # ── Vacuole training pair ────────────────────────────────────────────────
    if vacuole_map:
        vac_mask = _build_vacuole_mask(labels, vacuole_map, decisions)
        paths = _write_tiff_pair(image, vac_mask, stem, training_dir / "vacuoles")
        result["vacuoles"] = paths

    # ── Parasite training pair ───────────────────────────────────────────────
    para_mask = _build_parasite_mask(labels, decisions)
    paths = _write_tiff_pair(image, para_mask, stem, training_dir / "parasites")
    result["parasites"] = paths

    return result


def clear_training_data(
    annotations_dir: str | Path,
    mode: str = "both",
    clear_model: bool = False,
) -> int:
    """
    Delete StarDist training TIFF pairs and optionally the trained model(s).

    Parameters
    ----------
    annotations_dir : str or Path
    mode : 'vacuoles' | 'parasites' | 'both'
        Which training dataset to clear.
    clear_model : bool
        If True, also remove the stardist_model/ subdirectory for the
        selected mode(s).

    Returns
    -------
    n_deleted : int
        Total number of TIFF files deleted.
    """
    import shutil

    annotations_dir = Path(annotations_dir)
    training_dir = annotations_dir / "training_data"
    model_dir = annotations_dir / "stardist_model"

    modes = ["vacuoles", "parasites"] if mode == "both" else [mode]
    n_deleted = 0

    for m in modes:
        base = training_dir / m
        for sub in ("images", "masks"):
            sub_dir = base / sub
            if sub_dir.exists():
                for f in sub_dir.glob("*.tif"):
                    f.unlink()
                    n_deleted += 1
        if clear_model:
            m_model = model_dir / m
            if m_model.exists():
                shutil.rmtree(m_model)

    return n_deleted
