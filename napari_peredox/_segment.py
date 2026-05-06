"""
_segment.py — PV segmentation using Cellpose-SAM (cpsam)

Purpose
-------
Takes a 2-D fluorescence image and returns a labelled mask where each integer
region corresponds to one candidate parasitophorous vacuole (PV).

Two-stage pipeline:
  Stage 1 — Cellpose-SAM (cpsam)
    Cellpose 4 ships a SAM-based encoder ("cpsam") that achieves strong
    generalisation without any user training data.  Model weights are
    downloaded automatically from HuggingFace on first use and cached in
    ~/.cellpose/models/.  No account or token is required.

    The model returns a (H, W) integer array: 0 = background, each positive
    integer identifies one segment.

  Stage 2 — post-hoc classifier filter (optional)
    After the user has curated several sessions, a RandomForest classifier
    trained on morphological + intensity features can automatically reject
    segments that look like false positives before they reach the curation panel.
    This stage is a no-op until a trained classifier exists in the annotations dir.
"""

from __future__ import annotations

import numpy as np

# The Cellpose model is moderately expensive to load.
# We cache it module-globally so it is only loaded once per napari session.
_cellpose_model = None


def _get_model():
    """
    Load and cache the Cellpose-SAM model.

    On first call, downloads 'cpsam' weights from HuggingFace (cached to
    ~/.cellpose/models/) and wraps them in a CellposeModel instance.
    Subsequent calls return the cached model immediately.

    GPU is used automatically if torch detects a CUDA device.
    Returns the model instance.
    """
    global _cellpose_model
    if _cellpose_model is None:
        import torch
        from cellpose import models

        if torch.cuda.is_available():
            # Explicitly initialise the CUDA context in the calling thread
            # before handing off to Cellpose, so the model reliably lands on GPU
            # regardless of whether we are in the main thread or a worker thread.
            torch.cuda.init()
            torch.zeros(1, device="cuda")  # force context creation

        gpu = torch.cuda.is_available()
        _cellpose_model = models.CellposeModel(gpu=gpu, pretrained_model="cpsam")

    return _cellpose_model


def preload_model() -> str:
    """
    Load the Cellpose-SAM model if not already cached and return a status string.

    Call this from the main thread before launching a worker so that:
      1. The CUDA context is established in the main thread (more reliable).
      2. The device used is visible in the log before segmentation starts.
    """
    import torch

    model = _get_model()
    try:
        device = str(next(model.net.parameters()).device)
    except Exception:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if device.startswith("cuda"):
        mem_mb = torch.cuda.memory_allocated(0) / 1e6
        return f"Cellpose-SAM loaded on GPU ({device}) — {mem_mb:.0f} MB used"
    return "Cellpose-SAM loaded on CPU (no CUDA device found)"


def compute_threshold(
    channel: np.ndarray,
    method: str,
    manual_value: float = 0.0,
    percentile: float = 50.0,
) -> float:
    """
    Compute an intensity threshold for a single 2-D channel image.

    Parameters
    ----------
    channel : np.ndarray (H, W)
        Single-channel float image.
    method : str
        ``'none'``       — no threshold (returns 0, all pixels kept)
        ``'manual'``     — use ``manual_value`` directly
        ``'otsu'``       — Otsu's method (automatic bimodal threshold)
        ``'percentile'`` — threshold at the given percentile of non-zero pixels
    manual_value : float
        Used when method='manual'.  Should be in the same units as the image.
    percentile : float
        Used when method='percentile'.  E.g. 50 = median of non-zero pixels.

    Returns
    -------
    float
        Threshold value.  Pixels below this are considered background.
    """
    if method == "none":
        return 0.0
    if method == "manual":
        return float(manual_value)
    if method == "otsu":
        from skimage.filters import threshold_otsu

        # Otsu requires at least two distinct intensity values
        if channel.max() > channel.min():
            return float(threshold_otsu(channel))
        return 0.0
    if method == "percentile":
        # Compute percentile over non-zero pixels only so background
        # (camera offset) does not drag the threshold down.
        nonzero = channel[channel > 0]
        if nonzero.size == 0:
            return 0.0
        return float(np.percentile(nonzero, percentile))
    raise ValueError(
        f"Unknown threshold method {method!r}. "
        "Choose 'none', 'manual', 'otsu', or 'percentile'."
    )


def segment_pvs(
    image: np.ndarray,
    channel_index: int = 0,
    use_composite: bool = False,
    model_path: str | None = None,
    min_area_px: float = 0.0,
    max_area_px: float = 1e9,
    max_eccentricity: float = 0.95,
    min_solidity: float = 0.60,
    diameter: float | None = None,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
    threshold_method: str = "none",
    threshold_channel: int | None = None,
    threshold_value: float = 0.0,
    threshold_percentile: float = 50.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Run Cellpose-SAM on a 2-D fluorescence image and return an integer label array.

    An optional intensity threshold is applied to the image **before** cellpose
    runs: pixels below the threshold are zeroed out so that dim autofluorescent
    objects are never presented to the model.  Any cellpose segment that falls
    entirely within a zeroed region will not be detected.

    Parameters
    ----------
    image : np.ndarray
        2-D (H, W) or 3-D (H, W, C) float or uint array.
        If 3-D and use_composite=False, only channel_index is passed to cellpose.
    channel_index : int
        Which channel to pass to cellpose.  Ignored when use_composite=True.
    use_composite : bool
        If True, pass the channel-wise maximum projection to cellpose.
    model_path : str, optional
        Currently unused (reserved for future fine-tuned model support).
    min_area_px : float
        Discard segments whose pixel area is smaller than this (pixels²).
    max_area_px : float
        Discard segments whose pixel area is larger than this (pixels²).
    max_eccentricity : float
        Discard segments more elongated than this (0 = perfect circle, 1 = line).
        Parasites are roughly circular; autofluorescent debris is often elongated.
    min_solidity : float
        Discard segments less solid than this (0–1, where 1 = perfectly convex).
        Parasites are compact; fragmented or irregular objects score low.
    diameter : float, optional
        Expected parasite diameter in pixels.  None = auto-estimate.
    flow_threshold : float
        Cellpose flow error threshold.  Default 0.4.
    cellprob_threshold : float
        Cellpose cell probability threshold.  Default 0.0.
    threshold_method : str
        How to compute the intensity mask:
        ``'none'``       — no masking (default)
        ``'manual'``     — use threshold_value directly
        ``'otsu'``       — automatic Otsu threshold
        ``'percentile'`` — threshold at threshold_percentile of non-zero pixels
    threshold_channel : int, optional
        Which channel to compute the threshold on.
        None = use the same channel as channel_index (or the composite).
    threshold_value : float
        Explicit cutoff when threshold_method='manual'.
    threshold_percentile : float
        Percentile cutoff (0–100) when threshold_method='percentile'.

    Returns
    -------
    filtered_labels : np.ndarray (H, W) int32
        Label image after morphological filtering.  0 = background.
    raw_labels : np.ndarray (H, W) int32
        Label image direct from Cellpose before any filtering.
        Useful to display alongside filtered_labels to diagnose missed detections.
    filter_stats : dict
        Keys: ``total_raw``, ``rejected_area``, ``rejected_eccentricity``,
        ``rejected_solidity``, ``kept``.
    """
    # ── Step 1: extract the 2-D segmentation image ───────────────────────────

    if image.ndim == 2:
        seg_img = image.astype(np.float32)
    elif image.ndim == 3:
        if use_composite:
            seg_img = image.max(axis=-1).astype(np.float32)
        else:
            seg_img = image[..., channel_index].astype(np.float32)
    else:
        raise ValueError(f"Expected 2-D or 3-D array, got shape {image.shape}")

    # ── Step 2: intensity threshold mask ─────────────────────────────────────
    # Compute threshold on the requested channel, then zero out pixels below it.
    # This prevents cellpose from detecting segments in dim / autofluorescent
    # regions.  Skipped entirely when threshold_method='none'.
    if threshold_method != "none":
        if image.ndim == 3 and threshold_channel is not None:
            # Use a specific channel for thresholding (e.g. mCherry)
            thresh_src = image[..., threshold_channel].astype(np.float32)
        else:
            # Default: use the same image that goes into cellpose
            thresh_src = seg_img

        cutoff = compute_threshold(
            thresh_src,
            method=threshold_method,
            manual_value=threshold_value,
            percentile=threshold_percentile,
        )
        # Check what fraction of the image survives the threshold.
        mask = (thresh_src >= cutoff).astype(np.float32)
        pct_kept = 100.0 * float(mask.mean())

        if pct_kept < 5.0:
            # The threshold is so aggressive that < 5 % of pixels survive.
            # Applying it would leave a near-blank image that Cellpose cannot
            # segment.  Fall back to the unmasked image and record the warning.
            filter_stats_thresh = {
                "threshold_method": threshold_method,
                "threshold_cutoff": float(cutoff),
                "pct_kept": pct_kept,
                "threshold_skipped": True,
            }
        else:
            # Zero out pixels below the cutoff in the segmentation image.
            # Cellpose will not generate masks in fully-zeroed regions.
            seg_img = seg_img * mask
            filter_stats_thresh = {
                "threshold_method": threshold_method,
                "threshold_cutoff": float(cutoff),
                "pct_kept": pct_kept,
                "threshold_skipped": False,
            }
    else:
        filter_stats_thresh = {
            "threshold_method": "none",
            "threshold_cutoff": 0.0,
            "pct_kept": 100.0,
            "threshold_skipped": False,
        }

    # ── Step 3: run Cellpose-SAM ─────────────────────────────────────────────
    model = _get_model()
    masks, _, _ = model.eval(
        seg_img,
        channels=None,
        diameter=diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        normalize=True,
    )
    raw_labels = masks.astype(np.int32)

    # ── Step 4: morphological filtering ──────────────────────────────────────
    filtered_labels, filter_stats = _filter_by_morphology(
        raw_labels, min_area_px, max_area_px, max_eccentricity, min_solidity
    )

    filter_stats.update(filter_stats_thresh)
    return filtered_labels, raw_labels, filter_stats


def apply_classifier_filter(
    labels: np.ndarray,
    features,  # pd.DataFrame from _learning.extract_features()
    classifier,
) -> np.ndarray:
    """
    Remove PV candidates predicted as false positives by the learned classifier.

    This is the active-learning payoff: as the user curates more sessions, the
    classifier becomes better at automatically eliminating junk segments.

    Parameters
    ----------
    labels : np.ndarray (H, W) int32
        Label image from segment_pvs().
    features : pd.DataFrame
        One row per label, index = label id.
        Must contain at least some columns from _learning.FEATURE_COLS.
    classifier : fitted sklearn estimator (Pipeline with predict())
        Loaded from curated_features.joblib by _learning.load_classifier().

    Returns
    -------
    filtered_labels : np.ndarray (H, W) int32
        Labels predicted as non-PV (class 0) are zeroed out.
        Remaining labels are re-numbered consecutively from 1.
    """
    from ._learning import FEATURE_COLS

    out = labels.copy()

    # Only use feature columns that were present in training and exist here
    valid_cols = [c for c in FEATURE_COLS if c in features.columns]
    if not valid_cols:
        # If no matching feature columns exist, skip filtering rather than crash
        return out

    # Build the feature matrix; fill any missing values with 0
    X = features[valid_cols].fillna(0).values
    label_ids = features.index.values

    # Predict: 1 = true PV (keep), 0 = false positive (remove)
    preds = classifier.predict(X)

    # Zero out every pixel belonging to a rejected label
    reject_ids = label_ids[preds == 0]
    for rid in reject_ids:
        out[out == rid] = 0

    # Re-number remaining labels so IDs are consecutive (1, 2, 3, …)
    out = _relabel(out)
    return out


def group_by_vacuole(labels: np.ndarray, dilation_px: int = 5) -> dict:
    """
    Group individual parasite segments into vacuoles by proximity.

    Strategy: dilate every parasite mask by `dilation_px` pixels, then find
    connected components in the union.  Parasites whose dilated masks overlap
    are assumed to be inside the same vacuole (they are physically close).
    Parasites in separate vacuoles remain in separate connected components.

    Parameters
    ----------
    labels : np.ndarray (H, W) int32
        Parasite-level label image from segment_pvs().
    dilation_px : int
        How many pixels to dilate each parasite mask before clustering.
        Should be at least half the typical gap between parasites within
        one vacuole.  Default 5.

    Returns
    -------
    vacuole_map : dict[int, int]
        Maps each parasite label → vacuole ID (a positive integer).
        Parasites that share a vacuole ID are in the same vacuole.
    """
    from skimage.measure import label as sk_label
    from skimage.measure import regionprops
    from skimage.morphology import dilation, disk

    # Dilate the binary union of all parasite masks
    binary = labels > 0
    dilated = dilation(binary, disk(dilation_px))

    # Connected components of the dilated mask → one component per vacuole
    vacuole_labels = sk_label(dilated, connectivity=2)

    # For each parasite, read the vacuole ID at its centroid pixel
    vacuole_map = {}
    for rp in regionprops(labels):
        cy, cx = int(round(rp.centroid[0])), int(round(rp.centroid[1]))
        vacuole_map[rp.label] = int(vacuole_labels[cy, cx])

    return vacuole_map


def vacuole_label_image(labels: np.ndarray, vacuole_map: dict) -> np.ndarray:
    """
    Build a label image where every pixel belonging to a parasite is coloured
    by its vacuole ID rather than its individual parasite ID.

    The result lets napari render all parasites within one vacuole in the same
    colour, making it easy to verify that the grouping is correct.

    Parameters
    ----------
    labels : np.ndarray (H, W) int32
        Parasite-level label image from segment_pvs().
    vacuole_map : dict[int, int]
        Output of group_by_vacuole() — maps parasite label → vacuole ID.

    Returns
    -------
    np.ndarray (H, W) int32
        Label image with vacuole IDs.  Background is 0.
    """
    out = np.zeros_like(labels)
    for parasite_id, vacuole_id in vacuole_map.items():
        out[labels == parasite_id] = vacuole_id
    return out


# ── Internal helpers ──────────────────────────────────────────────────────────


def _filter_by_morphology(
    labels: np.ndarray,
    min_area_px: float,
    max_area_px: float,
    max_eccentricity: float,
    min_solidity: float,
) -> tuple[np.ndarray, dict]:
    """
    Zero out labels that fail any of three morphological criteria and return
    per-criterion rejection counts for diagnostic logging.

      1. Pixel area outside [min_area_px, max_area_px]
         — rejects objects that are too small or too large
      2. Eccentricity > max_eccentricity
         — rejects elongated / filamentous objects (parasites are roughly circular)
      3. Solidity < min_solidity
         — rejects fragmented or irregular objects (parasites are compact and convex)

    Any criterion can be effectively disabled by passing an extreme value:
      min_area_px=0, max_area_px=1e9 — disable size gate
      max_eccentricity=1.0            — disable eccentricity gate
      min_solidity=0.0                — disable solidity gate

    Returns
    -------
    filtered : np.ndarray (H, W) int32
    stats : dict  — keys: total_raw, rejected_area, rejected_eccentricity,
                           rejected_solidity, kept
    """
    from skimage.measure import regionprops

    out = labels.copy()
    n_area = n_ecc = n_sol = 0
    total = 0

    for rp in regionprops(labels):
        total += 1
        reject_reason = None
        if rp.area < min_area_px or rp.area > max_area_px:
            reject_reason = "area"
        elif rp.eccentricity > max_eccentricity:
            reject_reason = "eccentricity"
        elif rp.solidity < min_solidity:
            reject_reason = "solidity"

        if reject_reason == "area":
            n_area += 1
            out[out == rp.label] = 0
        elif reject_reason == "eccentricity":
            n_ecc += 1
            out[out == rp.label] = 0
        elif reject_reason == "solidity":
            n_sol += 1
            out[out == rp.label] = 0

    stats = {
        "total_raw": total,
        "rejected_area": n_area,
        "rejected_eccentricity": n_ecc,
        "rejected_solidity": n_sol,
        "kept": total - n_area - n_ecc - n_sol,
    }
    return out, stats


def _relabel(labels: np.ndarray) -> np.ndarray:
    """
    Re-number label IDs to be consecutive integers starting from 1.

    After removing some labels, IDs may be non-consecutive (e.g. 1, 3, 7).
    This function converts the binary mask back to consecutive labels so that
    'labels.max()' equals the true number of segments.
    """
    from skimage.measure import label as sk_label

    # Convert to binary (any non-zero = foreground) then re-label
    binary = labels > 0
    return sk_label(binary, connectivity=2).astype(np.int32)
