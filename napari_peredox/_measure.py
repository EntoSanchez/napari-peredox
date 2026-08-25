"""
_measure.py — Fluorescence measurements for each segmented PV

Purpose
-------
For every labelled region in the segmentation output, compute:
  - Pixel area
  - Per-channel mean intensity and integrated density (sum of pixel values)
  - The primary output: cpTSapphire / mCherry integrated-density ratio

Background
----------
Peredox is a ratiometric sensor.  The relevant quantity is:

    ratio = Σ(cpTSapphire pixels inside PV) / Σ(mCherry pixels inside PV)

This is identical to mean_cptsa / mean_mcherry because area cancels,
but storing the raw integrated densities (= RawIntDen in Fiji) allows the
user to verify their measurements against ImageJ workflows.

The word "integrated density" comes from Fiji/ImageJ terminology:
  - RawIntDen  = sum of all pixel values inside the ROI  ← what we compute
  - IntDen     = RawIntDen × pixel area (µm²)            ← only if calibrated
  - Mean       = RawIntDen / area_px

We use RawIntDen because it is invariant to pixel size calibration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from skimage.measure import regionprops


def measure_pvs(
    labels: np.ndarray,
    image: np.ndarray,
    ch_cptsa: int,
    ch_mcherry: int,
    ch_names: dict[int, str] | None = None,
    pixel_size_um: float | None = None,
) -> pd.DataFrame:
    """
    Measure fluorescence properties for every labelled PV.

    Parameters
    ----------
    labels : np.ndarray (H, W) int32
        Label image from segmentation.  0 = background.
    image : np.ndarray (H, W, C) float32
        Multi-channel image with channels on the last axis.
    ch_cptsa : int
        Channel index for cpTSapphire.
    ch_mcherry : int
        Channel index for mCherry.
    ch_names : dict {int → str}, optional
        Human-readable name for each channel index.
        Used as column name suffixes (e.g. {0: 'cptsa', 1: 'mcherry'}).
        Defaults to 'ch0', 'ch1', etc.
    pixel_size_um : float, optional
        Physical size of one pixel in µm.
        If provided, adds an `area_um2` column (area_px × pixel_size²).
        Leave None to work purely in pixel units.

    Returns
    -------
    df : pd.DataFrame, indexed by label id
        Columns (example for 2-channel image with ch_names={0:'cptsa',1:'mcherry'}):
          centroid_y, centroid_x   — pixel coordinates of the segment centroid
          area_px                  — segment area in pixels
          area_um2                 — physical area (only if pixel_size_um provided)
          mean_cptsa               — mean cpTSapphire intensity
          intden_cptsa             — integrated density (RawIntDen) cpTSapphire
          mean_mcherry             — mean mCherry intensity
          intden_mcherry           — integrated density mCherry
          ratio_cptsa_mcherry      — primary Peredox readout
    """
    # ── Step 1: resolve channel names ────────────────────────────────────────
    if ch_names is None:
        ch_names = {}

    # Make sure the two key channels have meaningful names
    cptsa_name = ch_names.get(ch_cptsa, f"ch{ch_cptsa}")
    mcherry_name = ch_names.get(ch_mcherry, f"ch{ch_mcherry}")

    # ── Step 2: ensure image is always (H, W, C) ─────────────────────────────
    n_channels = image.shape[-1] if image.ndim == 3 else 1
    if image.ndim == 2:
        image = image[..., np.newaxis]  # add a dummy channel axis

    # Total image area used for area_fraction computation
    image_area_px = float(labels.shape[0] * labels.shape[1])

    # ── Step 3: iterate over all labelled regions ─────────────────────────────
    rows = []
    for rp in regionprops(labels):
        # `rp.slice` is a (row_slice, col_slice) bounding box.
        # We crop both the label array and the image to this box,
        # then build a boolean mask for just this label's pixels.
        mask = labels[rp.slice] == rp.label

        # ── Shape / morphology ────────────────────────────────────────────────
        perimeter = rp.perimeter if rp.perimeter > 0 else 1.0
        circularity = (4.0 * np.pi * rp.area) / (perimeter**2)

        major = rp.axis_major_length
        minor = rp.axis_minor_length
        aspect_ratio = major / minor if minor > 0 else np.nan
        roundness = (4.0 * rp.area) / (np.pi * major**2) if major > 0 else np.nan

        row: dict = {
            "label": rp.label,
            # Centroid coordinates in the full image frame
            "centroid_y": rp.centroid[0],
            "centroid_x": rp.centroid[1],
            # Size
            "area_px": rp.area,
            "area_fraction": rp.area / image_area_px,
            # Shape descriptors (mirrors ImageJ Analyze > Set Measurements)
            "perimeter": rp.perimeter,
            "circularity": circularity,  # 4π·A / P²  (1 = circle)
            "aspect_ratio": aspect_ratio,  # major / minor axis
            "roundness": roundness,  # 4A / (π · major²)
            "eccentricity": rp.eccentricity,  # 0 = circle, 1 = line
            "solidity": rp.solidity,  # area / convex_hull_area
            "extent": rp.extent,  # area / bounding_box_area
            # Ellipse fit
            "major_axis_length": major,
            "minor_axis_length": minor,
            "orientation": rp.orientation,  # radians, major axis vs x-axis
        }

        # Optional physical area (µm²)
        if pixel_size_um is not None:
            row["area_um2"] = rp.area * (pixel_size_um**2)

        # ── Step 4: compute per-channel intensity statistics ─────────────────
        for ch in range(n_channels):
            ch_name = ch_names.get(ch, f"ch{ch}")

            # Crop the channel to the bounding box, then flatten to just the
            # pixels inside the mask (excludes background within the bounding box)
            patch = image[rp.slice][..., ch].astype(np.float64)
            pixels_in_mask = patch[mask]

            row[f"mean_{ch_name}"] = float(pixels_in_mask.mean())
            row[f"median_{ch_name}"] = float(np.median(pixels_in_mask))
            row[f"std_{ch_name}"] = float(pixels_in_mask.std())
            row[f"min_{ch_name}"] = float(pixels_in_mask.min())
            row[f"max_{ch_name}"] = float(pixels_in_mask.max())
            # Integrated density = sum of all pixel values (= RawIntDen in Fiji)
            row[f"intden_{ch_name}"] = float(pixels_in_mask.sum())
            # Modal grey value: most frequent intensity bin (256 bins over 0–max)
            px_max = pixels_in_mask.max()
            if px_max > 0:
                counts, edges = np.histogram(
                    pixels_in_mask, bins=256, range=(0, px_max)
                )
                modal_bin = int(counts.argmax())
                row[f"mode_{ch_name}"] = float(
                    0.5 * (edges[modal_bin] + edges[modal_bin + 1])
                )
            else:
                row[f"mode_{ch_name}"] = 0.0
            # Distribution shape (skewness = 0 for symmetric, kurtosis = 0 for normal)
            if len(pixels_in_mask) >= 4:
                row[f"skewness_{ch_name}"] = float(scipy_stats.skew(pixels_in_mask))
                row[f"kurtosis_{ch_name}"] = float(scipy_stats.kurtosis(pixels_in_mask))
            else:
                row[f"skewness_{ch_name}"] = np.nan
                row[f"kurtosis_{ch_name}"] = np.nan

        # ── Step 5: compute the Peredox ratio three ways ─────────────────────
        intden_cptsa = row[f"intden_{cptsa_name}"]
        intden_mcherry = row[f"intden_{mcherry_name}"]
        mean_cptsa = row[f"mean_{cptsa_name}"]
        mean_mcherry = row[f"mean_{mcherry_name}"]
        median_cptsa = row[f"median_{cptsa_name}"]
        median_mcherry = row[f"median_{mcherry_name}"]

        row["ratio_intden"] = (
            intden_cptsa / intden_mcherry if intden_mcherry != 0 else np.nan
        )
        row["ratio_mean"] = mean_cptsa / mean_mcherry if mean_mcherry != 0 else np.nan
        row["ratio_median"] = (
            median_cptsa / median_mcherry if median_mcherry != 0 else np.nan
        )
        # Keep old column name as alias for the intden ratio so existing code
        # that references ratio_cptsa_mcherry still works.
        row["ratio_cptsa_mcherry"] = row["ratio_intden"]

        rows.append(row)

    # ── Step 6: assemble into a DataFrame ────────────────────────────────────
    if not rows:
        # No segments found — return an empty DataFrame
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("label")
    return df


def select_one_per_vacuole(
    df: pd.DataFrame,
    vacuole_map: dict,
    method: str = "largest",
) -> pd.DataFrame:
    """
    Reduce a per-parasite measurements DataFrame to one row per vacuole.

    Parasites within one vacuole share the same vacuole ID in `vacuole_map`.
    This function picks a single representative parasite per vacuole so that
    the dead space between parasites (visible in the MIP) does not dilute
    mean intensity measurements.

    Parameters
    ----------
    df : pd.DataFrame
        Output of measure_pvs(), indexed by parasite label.
    vacuole_map : dict {int → int}
        Maps parasite label → vacuole ID.  From _segment.group_by_vacuole().
    method : str
        How to pick the representative parasite from each vacuole:
        - ``'largest'``       — parasite with the greatest area_px
        - ``'highest_ratio'`` — parasite with the highest cpTSapphire/mCherry ratio
        - ``'median_ratio'``  — parasite closest to the median ratio in the group
        - ``'mean_ratio'``    — parasite closest to the mean ratio in the group

    Returns
    -------
    pd.DataFrame
        Same columns as the input, but with at most one row per vacuole.
        A ``vacuole_id`` column is added so downstream code can trace groupings.
        Index values are the selected parasite labels (unchanged).
    """
    if df.empty:
        return df

    # Tag each row with its vacuole ID
    tagged = df.copy()
    tagged["vacuole_id"] = tagged.index.map(vacuole_map)

    selected_rows = []
    chosen_labels = []  # track label IDs so we can restore the named index
    for _vac_id, group in tagged.groupby("vacuole_id"):
        if method == "largest":
            chosen_label = group["area_px"].idxmax()
        elif method == "highest_ratio":
            chosen_label = group["ratio_cptsa_mcherry"].idxmax()
        elif method == "median_ratio":
            ref = group["ratio_cptsa_mcherry"].median()
            chosen_label = (group["ratio_cptsa_mcherry"] - ref).abs().idxmin()
        elif method == "mean_ratio":
            ref = group["ratio_cptsa_mcherry"].mean()
            chosen_label = (group["ratio_cptsa_mcherry"] - ref).abs().idxmin()
        else:
            raise ValueError(
                f"Unknown method {method!r}. "
                "Choose 'largest', 'highest_ratio', 'median_ratio', or 'mean_ratio'."
            )
        row = group.loc[chosen_label].copy()
        row["parasites_per_vacuole"] = len(group)
        row["vacuole_area_px"] = float(group["area_px"].sum())
        # Per-vacuole ratio aggregates from individual parasite measurements
        if "ratio_cptsa_mcherry" in group.columns:
            row["mean_parasite_ratio"] = float(group["ratio_cptsa_mcherry"].mean())
            row["median_parasite_ratio"] = float(group["ratio_cptsa_mcherry"].median())
        selected_rows.append(row)
        chosen_labels.append(chosen_label)

    result = pd.DataFrame(selected_rows)
    result.index = chosen_labels
    result.index.name = "label"
    return result


def _snap_pow2(ratio: float) -> int:
    """Round a continuous parasite-count estimate to the nearest power of 2 (≥ 1)."""
    if ratio <= 0:
        return 1
    return max(1, 2 ** int(round(np.log2(max(ratio, 1.0)))))


def add_estimated_parasites(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ``estimated_parasites_pow2`` column to a per-vacuole measurements table.

    Uses the median vacuole_area_px of singletons (parasites_per_vacuole == 1)
    as the reference area for one parasite, then snaps each PV's area ratio to
    the nearest power of 2 (1, 2, 4, 8, 16 …).

    No-ops if the required columns are absent or there are no singletons.
    """
    if "vacuole_area_px" not in df.columns or "parasites_per_vacuole" not in df.columns:
        return df
    singles = df.loc[df["parasites_per_vacuole"] == 1, "vacuole_area_px"]
    if singles.empty:
        return df
    area_per_parasite = float(singles.median())
    out = df.copy()
    out["estimated_parasites_pow2"] = (
        out["vacuole_area_px"]
        .apply(lambda a: _snap_pow2(float(a) / area_per_parasite))
        .astype(int)
    )
    return out


def measure_vacuoles(
    labels: np.ndarray,
    image: np.ndarray,
    vacuole_map: dict[int, int],
    ch_cptsa: int,
    ch_mcherry: int,
    ch_names: dict[int, str] | None = None,
    pixel_size_um: float | None = None,
) -> pd.DataFrame:
    """
    Measure fluorescence for entire vacuoles (union of all parasite pixels).

    Remaps the per-parasite label image to per-vacuole labels and runs the
    same regionprops pipeline as measure_pvs().  Returns a DataFrame indexed
    by vacuole_id with the same column set as measure_pvs().

    Parameters
    ----------
    labels : np.ndarray (H, W)
        Per-parasite label image (0 = background).
    image : np.ndarray (H, W, C)
        Multi-channel image.
    vacuole_map : dict {parasite_label → vacuole_id}
        From _segment.group_by_vacuole().
    ch_cptsa, ch_mcherry, ch_names, pixel_size_um
        Same meaning as in measure_pvs().

    Returns
    -------
    pd.DataFrame indexed by vacuole_id.
    """
    vac_labels = np.zeros_like(labels)
    for para_label, vac_id in vacuole_map.items():
        vac_labels[labels == para_label] = vac_id
    return measure_pvs(vac_labels, image, ch_cptsa, ch_mcherry, ch_names, pixel_size_um)


def summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mean, std, median, and n across all PVs in df.

    Useful for a quick per-image summary after curation.
    Operates only on numeric columns (ignores string metadata).

    Parameters
    ----------
    df : pd.DataFrame
        Output of measure_pvs(), possibly filtered to accepted PVs only.

    Returns
    -------
    summary : pd.DataFrame
        Columns: mean, std, median, n
        Rows: one per original numeric column
    """
    # Select only numeric columns; skips centroid coords if not meaningful
    numeric = df.select_dtypes(include="number")

    summary = pd.DataFrame(
        {
            "mean": numeric.mean(),
            "std": numeric.std(),
            "median": numeric.median(),
            "n": numeric.count(),
        }
    )
    return summary
