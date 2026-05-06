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
    csv_path = annotations_dir / "curated_features.csv"

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
