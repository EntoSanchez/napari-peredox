"""
_learning.py — Active learning: feature extraction and classifier training

Purpose
-------
This module implements the "learn over time" capability of the plugin.

How it works end-to-end:
  1.  The user curates a set of cellSAM segments as true PV (accepted=1) or
      false positive (accepted=0) via the CurationWidget.
  2.  `_io.append_curated_annotations()` appends these decisions, together with
      the morphological + intensity features extracted here, to a running CSV
      file (`annotations/curated_features.csv`).
  3.  `train_classifier()` fits a RandomForestClassifier on all accumulated rows.
      The trained model is serialised as a .joblib file alongside the CSV.
  4.  On the next segmentation run, `_widget.py` calls `load_classifier()`,
      then `_segment.apply_classifier_filter()` uses the model to automatically
      remove segments that look like false positives.

The model is intentionally simple (RandomForest on hand-crafted features) so
that:
  - Training is instant (<1 s) even for hundreds of examples
  - No GPU is required
  - Feature importances are human-inspectable
  - It starts working with as few as 10 annotated examples

FEATURE_COLS defines which features are used for classification.  Add new
feature names here and to extract_features() to expand the model's vocabulary.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from skimage.measure import regionprops

# ── Feature column registry ───────────────────────────────────────────────────
# This list is the contract between extract_features() (which produces them),
# train_classifier() (which fits on them), and apply_classifier_filter()
# (which applies the model).  Order does not matter; missing columns are filled
# with 0 rather than raising an error.

FEATURE_COLS = [
    "area_px",  # Pixel count — PVs tend to occupy a characteristic size range
    "eccentricity",  # Shape elongation (0 = circle, 1 = line) — PVs are roughly round
    "solidity",  # area / convex_hull_area — PVs are solid; debris is irregular
    "extent",  # area / bounding_box_area — compact objects score high
    "perimeter",  # Boundary length in pixels — correlated with area but adds shape info
    "mean_intensity_seg_ch",  # Mean brightness on segmentation channel — helps separate signal from noise
    "std_intensity_seg_ch",  # Intensity variance — uniform interiors vs. textured/noisy regions
    "mean_cptsa",  # Mean cpTSapphire intensity — true PVs are bright in this channel
    "mean_mcherry",  # Mean mCherry intensity — true PVs have mCherry signal too
    "intden_cptsa",  # Integrated density cpTSapphire — correlated with PV size × brightness
    "intden_mcherry",  # Integrated density mCherry
    "ratio_cptsa_mcherry",  # The Peredox ratio itself — may differ between PVs and debris
]


def extract_features(
    labels: np.ndarray,
    image: np.ndarray,
    seg_channel: int = 0,
    ch_cptsa: int = 0,
    ch_mcherry: int = 1,
    ch_names: dict[int, str] | None = None,
) -> pd.DataFrame:
    """
    Extract morphological and intensity features for every label in `labels`.

    These features are used both for:
      - Training the classifier (after curation)
      - Applying the trained classifier (at inference time)

    Parameters
    ----------
    labels : np.ndarray (H, W) int32
        Segmentation label image.
    image : np.ndarray (H, W, C) float32
        Multi-channel image.
    seg_channel : int
        Which channel was used for segmentation (used for intensity features).
    ch_cptsa : int
        Channel index of cpTSapphire.
    ch_mcherry : int
        Channel index of mCherry.
    ch_names : dict {int → str}, optional
        Same channel name mapping used in _measure.py.

    Returns
    -------
    df : pd.DataFrame, indexed by label id
        One row per segment, columns matching FEATURE_COLS as available.
    """
    if ch_names is None:
        ch_names = {}

    # Resolve the string names that will appear in column headers
    cptsa_name = ch_names.get(ch_cptsa, f"ch{ch_cptsa}")
    mcherry_name = ch_names.get(ch_mcherry, f"ch{ch_mcherry}")

    # Ensure image always has a channel axis
    if image.ndim == 2:
        image = image[..., np.newaxis]

    # The segmentation channel is used to compute per-pixel intensity features
    seg_img = image[..., seg_channel].astype(np.float64)

    rows = []
    # regionprops with an intensity_image gives us both shape and intensity stats
    for rp in regionprops(labels, intensity_image=seg_img):
        # Boolean mask for this label within its bounding box
        mask = labels[rp.slice] == rp.label

        # Pixel values from the segmentation channel inside the mask
        patch_seg = seg_img[rp.slice][mask]

        # ── Morphological features ────────────────────────────────────────────
        # These are computed by skimage.measure.regionprops automatically;
        # we just copy them out.
        row: dict = {
            "label": rp.label,
            "centroid_y": rp.centroid[
                0
            ],  # saved for dilation calibration, not a classifier feature
            "centroid_x": rp.centroid[1],
            "area_px": rp.area,
            "eccentricity": rp.eccentricity,  # 0 = perfect circle
            "solidity": rp.solidity,  # area / convex hull area
            "extent": rp.extent,  # area / bounding rect area
            "perimeter": rp.perimeter,
            # Intensity stats on the segmentation channel
            "mean_intensity_seg_ch": float(patch_seg.mean()),
            "std_intensity_seg_ch": float(patch_seg.std()),
        }

        # ── Per-channel intensity features ────────────────────────────────────
        # For each named channel (cpTSapphire, mCherry, any others) compute
        # mean intensity and integrated density inside the mask.
        for ch_idx, ch_name in ch_names.items():
            if ch_idx < image.shape[-1]:
                patch = image[rp.slice][..., ch_idx][mask].astype(np.float64)
                row[f"mean_{ch_name}"] = float(patch.mean())
                row[f"intden_{ch_name}"] = float(patch.sum())

        # ── Ratio feature ─────────────────────────────────────────────────────
        # The Peredox ratio is itself a discriminative feature: debris tends to
        # have random or extreme ratios, while true PVs cluster in a biologically
        # meaningful range.
        intden_cptsa = row.get(f"intden_{cptsa_name}", np.nan)
        intden_mcherry = row.get(f"intden_{mcherry_name}", np.nan)

        if intden_mcherry and intden_mcherry != 0:
            row["ratio_cptsa_mcherry"] = intden_cptsa / intden_mcherry
        else:
            row["ratio_cptsa_mcherry"] = np.nan

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).set_index("label")


def train_classifier(csv_path: str | Path) -> object | None:
    """
    Fit a RandomForestClassifier on all annotated examples in csv_path.

    The trained model is saved as a .joblib file adjacent to the CSV so it
    can be loaded automatically in future sessions via load_classifier().

    Parameters
    ----------
    csv_path : str or Path
        Path to curated_features.csv (written by _io.append_curated_annotations).
        Must contain an `accepted` column (1 = true PV, 0 = false positive).

    Returns
    -------
    clf : fitted sklearn Pipeline, or None
        Returns None if there are fewer than 10 annotated examples or if only
        one class is represented (can't fit a binary classifier with one class).
    """
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    csv_path = Path(csv_path)
    if not csv_path.exists():
        return None

    # ── Step 1: load and validate the annotation data ────────────────────────
    df = pd.read_csv(csv_path)

    if "accepted" not in df.columns:
        return None

    # Drop rows where the user skipped (accepted == -1 or NaN)
    df = df.dropna(subset=["accepted"])

    # Minimum requirements for a meaningful model
    if len(df) < 10:
        return None  # Too little data — wait for more curation sessions
    if df["accepted"].nunique() < 2:
        return None  # Only one class — can't train a binary classifier

    # ── Step 2: build feature matrix ─────────────────────────────────────────
    # Use only columns that exist in FEATURE_COLS; others are ignored.
    valid_cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[valid_cols].fillna(0).values  # fill NaN ratios etc. with 0
    y = df["accepted"].astype(int).values

    # ── Step 3: build and fit the pipeline ───────────────────────────────────
    # StandardScaler ensures features on different scales (area vs. ratio)
    # don't dominate the RandomForest splits.
    # class_weight='balanced' compensates for imbalanced accept/reject counts.
    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=100,  # 100 trees — fast to train, stable predictions
                    max_depth=8,  # limit depth to reduce overfitting on small data
                    class_weight="balanced",  # weight minority class up
                    random_state=42,  # reproducible results
                ),
            ),
        ]
    )
    clf.fit(X, y)

    # ── Step 4: serialise the model ──────────────────────────────────────────
    # Save alongside the CSV so load_classifier() can find it automatically
    model_path = csv_path.with_suffix(".joblib")
    joblib.dump(clf, model_path)

    return clf


def load_classifier(annotations_dir: str | Path) -> object | None:
    """
    Load a previously trained classifier from the annotations directory.

    Returns None if no model file is found (i.e. classifier has never been
    trained or the annotations dir is new).
    """
    import joblib

    model_path = Path(annotations_dir) / "curated_features.joblib"
    if model_path.exists():
        return joblib.load(model_path)
    return None


def learn_dilation_radius(csv_path: str | Path, default_px: int = 5) -> int:
    """
    Estimate the optimal dilation radius for vacuole grouping from curated data.

    Uses the `vacuole_id` assignments saved during curation (user-confirmed or
    auto-assigned) together with parasite centroid coordinates to compute the
    typical within-vacuole gap.  Returns half the median within-vacuole centroid
    distance, which is the minimum dilation needed to bridge that gap.

    Parameters
    ----------
    csv_path : str or Path
        Path to curated_features.csv.  Must contain `vacuole_id`, `centroid_y`,
        `centroid_x`, and `image_stem` columns.
    default_px : int
        Value returned when there are not enough paired parasites to calibrate.

    Returns
    -------
    int
        Recommended dilation radius in pixels (at least 1).
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return default_px

    df = pd.read_csv(csv_path)
    required = {"vacuole_id", "centroid_y", "centroid_x", "image_stem"}
    if not required.issubset(df.columns):
        return default_px

    # Only consider accepted parasites with valid vacuole IDs
    if "accepted" in df.columns:
        df = df[df["accepted"] == 1].copy()
    df = df.dropna(subset=["vacuole_id", "centroid_y", "centroid_x"])
    if df.empty:
        return default_px

    # Compute pairwise centroid distances for all within-vacuole parasite pairs
    all_gaps: list[float] = []
    for (_stem, _vac_id), group in df.groupby(["image_stem", "vacuole_id"]):
        if len(group) < 2:
            continue
        ys = group["centroid_y"].values
        xs = group["centroid_x"].values
        for i in range(len(ys)):
            for j in range(i + 1, len(ys)):
                dist = float(np.sqrt((ys[i] - ys[j]) ** 2 + (xs[i] - xs[j]) ** 2))
                all_gaps.append(dist)

    if not all_gaps:
        return default_px

    # Dilation should bridge at least half the median within-vacuole gap
    median_gap = float(np.median(all_gaps))
    return max(1, int(np.ceil(median_gap / 2.0)))


def classifier_stats(csv_path: str | Path) -> dict:
    """
    Return a summary of the annotation dataset for display in the status panel.

    Returns a dict with keys: total, accepted, rejected.
    All values are 0 if the CSV does not exist or has no annotated rows.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return {"total": 0, "accepted": 0, "rejected": 0}

    df = pd.read_csv(csv_path)
    if "accepted" not in df.columns:
        return {"total": 0, "accepted": 0, "rejected": 0}

    # Exclude skipped rows (accepted == -1 or NaN)
    df = df.dropna(subset=["accepted"])

    n_accept = int((df["accepted"] == 1).sum())
    n_reject = int((df["accepted"] == 0).sum())
    return {"total": len(df), "accepted": n_accept, "rejected": n_reject}
