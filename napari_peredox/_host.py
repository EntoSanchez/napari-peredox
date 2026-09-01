"""
_host.py — Host-cell segmentation and infection analysis

Purpose
-------
Implements the host-cell analysis mode (spec:
docs/superpowers/specs/2026-08-25-host-analysis-design.md):

  Stage H1  segment_host_cells() — cpSAM on the host channel with bright
            pixels clipped first, so very bright intracellular parasites do
            not compress the host cells' dynamic range during cpSAM's
            intensity normalisation.
  Stage H3  assign_to_hosts()   — majority-overlap parasite→host assignment.
            measure_hosts()     — per-host Peredox ratio on the cytosol
            (host mask minus dilated parasite pixels) plus infection
            metadata columns.

Stage H2 (parasite detection inside accepted hosts) reuses the existing PV
pipeline in _segment.py on a host-masked image and needs no code here.

Population rule (spec §2): every accepted fluorescent host is measured,
infected or not; parasites are only analysed inside accepted fluorescent
hosts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Columns in measure_hosts() output that only make sense when parasites were
# actually assessed.  Host-only analysis (no parasite stage) drops these from
# its exported tables so a reader can never mistake "never checked" for
# "verified uninfected".
INFECTION_COLS = [
    "infected",
    "n_parasites",
    "n_vacuoles",
    "parasite_area_px",
    "cytosol_empty",
]


def drop_infection_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *df* without the parasite-assessment columns."""
    return df.drop(columns=[c for c in INFECTION_COLS if c in df.columns])


def clip_bright(channel: np.ndarray, percentile: float = 99.0) -> np.ndarray:
    """
    Clip a single-channel image at the given percentile of its non-zero pixels.

    Used before cpSAM host segmentation: parasites are far brighter than the
    host cytosol, and without clipping they dominate cpSAM's intensity
    normalisation, flattening host contrast.

    Parameters
    ----------
    channel : np.ndarray (H, W)
        Single-channel image (any numeric dtype).
    percentile : float
        Percentile (0–100) of non-zero pixel values used as the clip ceiling.
        Values >= 100 disable clipping.

    Returns
    -------
    np.ndarray (H, W) float32
        Clipped copy. All-zero input or percentile >= 100 returns an
        unmodified float32 copy.
    """
    out = np.asarray(channel, dtype=np.float32).copy()
    if percentile >= 100.0:
        return out
    nonzero = out[out > 0]
    if nonzero.size == 0:
        return out
    cutoff = np.float32(np.percentile(nonzero, percentile))
    np.clip(out, None, cutoff, out=out)
    return out


def assign_to_hosts(
    para_labels: np.ndarray,
    host_labels: np.ndarray,
    vacuole_map: dict[int, int] | None = None,
) -> tuple[dict[int, int], dict[int, int], list[int]]:
    """
    Assign each parasite (and vacuole) to the host cell it overlaps most.

    Majority pixel overlap decides ownership.  np.bincount + argmax makes the
    tie-break deterministic: equal counts resolve to the lowest index, i.e.
    background (0) first, then the lowest host ID.  A parasite whose plurality
    value is background is *dropped* — after Stage H2 masking this can only
    happen at mask edges following curation redraws.

    Parameters
    ----------
    para_labels : np.ndarray (H, W) int32
        Parasite label image (0 = background).
    host_labels : np.ndarray (H, W) int32
        Accepted host label image (0 = background).
    vacuole_map : dict {parasite_label → vacuole_id}, optional
        Stage 2 grouping.  When given, each vacuole is assigned to the host
        holding the majority of its member-parasite pixels.

    Returns
    -------
    para_to_host : dict {parasite_label → host_id}
    vac_to_host : dict {vacuole_id → host_id}   (empty if vacuole_map is None)
    dropped : list[int]
        Parasite labels whose plurality pixel was background.
    """
    from collections import defaultdict

    from skimage.measure import regionprops

    para_to_host: dict[int, int] = {}
    dropped: list[int] = []
    # Per-vacuole pixel counts per host, accumulated across member parasites
    vac_counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))

    for rp in regionprops(para_labels):
        mask = para_labels[rp.slice] == rp.label
        host_vals = host_labels[rp.slice][mask]
        counts = np.bincount(host_vals)
        winner = int(np.argmax(counts))
        if winner == 0:
            dropped.append(int(rp.label))
        else:
            para_to_host[int(rp.label)] = winner

        if vacuole_map is not None:
            vac_id = vacuole_map.get(int(rp.label))
            if vac_id is not None:
                for hid in np.nonzero(counts)[0]:
                    if hid != 0:
                        vac_counts[int(vac_id)][int(hid)] += int(counts[hid])

    vac_to_host: dict[int, int] = {}
    for vac_id, cmap in vac_counts.items():
        # max count wins; ties resolve to the lowest host ID
        best_host = min(cmap, key=lambda hid: (-cmap[hid], hid))
        vac_to_host[vac_id] = best_host

    return para_to_host, vac_to_host, dropped


def measure_hosts(
    host_labels: np.ndarray,
    para_labels: np.ndarray,
    image: np.ndarray,
    para_to_host: dict[int, int],
    vac_to_host: dict[int, int],
    dilation_px: int = 3,
    exclude_labels: np.ndarray | None = None,
    ch_cptsa: int = 0,
    ch_mcherry: int = 1,
    ch_names: dict[int, str] | None = None,
    pixel_size_um: float | None = None,
) -> pd.DataFrame:
    """
    Measure Peredox fluorescence per host cell on the parasite-free cytosol.

    The cytosol mask is the host mask minus `exclude_labels` (or the parasite
    labels when none is given) dilated by `dilation_px` — a buffer against
    signal bleed-over.  Host+parasites mode passes the Stage-1 vacuole masks
    so the whole PV lumen is removed, not just the parasite bodies.  Ratios come
    from measure_pvs() run on that cytosol label image; host label IDs are
    preserved, so the returned index is the host ID.

    Every host in `host_labels` gets exactly one row.  A host whose cytosol
    vanishes entirely (heavy infection) keeps its row with NaN measurement
    columns and cytosol_empty=True.

    Added metadata columns (spec §3.1): infected, n_parasites, n_vacuoles,
    parasite_area_px, host_area_px_total, on_border, cytosol_empty.
    """
    from skimage.measure import regionprops
    from skimage.morphology import dilation, disk

    from ._measure import measure_pvs

    host_ids = sorted(int(v) for v in np.unique(host_labels) if v != 0)

    # ── Cytosol: host minus dilated parasite pixels ──────────────────────────
    # What gets subtracted from the host: the whole vacuole mask when the
    # caller supplies one (mCherry fills the entire PV lumen, so the dead space
    # between parasites still carries parasite-derived signal), otherwise just
    # the parasite bodies.
    exclude_src = para_labels if exclude_labels is None else exclude_labels
    cytosol = host_labels.copy()
    excluded_mask = np.zeros(host_labels.shape, dtype=bool)
    if exclude_src is not None and exclude_src.max() > 0:
        excluded_mask = exclude_src > 0
        if dilation_px > 0:
            excluded_mask = dilation(excluded_mask, disk(dilation_px))
        cytosol[excluded_mask] = 0

    df = measure_pvs(cytosol, image, ch_cptsa, ch_mcherry, ch_names, pixel_size_um)

    # ── Full-footprint stats and border flags ────────────────────────────────
    H, W = host_labels.shape
    total_area: dict[int, float] = {}
    on_border: dict[int, bool] = {}
    for rp in regionprops(host_labels):
        total_area[int(rp.label)] = float(rp.area)
        minr, minc, maxr, maxc = rp.bbox
        on_border[int(rp.label)] = minr == 0 or minc == 0 or maxr == H or maxc == W

    # ── Parasite burden per host ─────────────────────────────────────────────
    n_para = dict.fromkeys(host_ids, 0)
    para_area = dict.fromkeys(host_ids, 0.0)
    for pid, hid in para_to_host.items():
        if hid in n_para:
            n_para[hid] += 1
            para_area[hid] += float((para_labels == pid).sum())
    n_vac = dict.fromkeys(host_ids, 0)
    for _vid, hid in vac_to_host.items():
        if hid in n_vac:
            n_vac[hid] += 1

    # ── Re-add hosts whose cytosol vanished entirely ─────────────────────────
    empty_ids = [h for h in host_ids if df.empty or h not in df.index]
    if empty_ids:
        nan_rows = pd.DataFrame(index=pd.Index(empty_ids, name="label"))
        df = pd.concat([df, nan_rows]) if not df.empty else nan_rows
    if "ratio_intden" not in df.columns:
        # Every host was fully covered — measure_pvs returned an empty frame,
        # so the ratio columns never materialised.  NaN keeps the schema.
        df["ratio_intden"] = np.nan
    if "ratio_mean" not in df.columns:
        df["ratio_mean"] = np.nan
    if "ratio_median" not in df.columns:
        df["ratio_median"] = np.nan
    df = df.loc[sorted(int(i) for i in df.index)]

    # ── Metadata columns ─────────────────────────────────────────────────────
    empty_set = set(empty_ids)
    df["cytosol_empty"] = [h in empty_set for h in df.index]
    df["infected"] = [n_para.get(h, 0) > 0 for h in df.index]
    df["n_parasites"] = [n_para.get(h, 0) for h in df.index]
    df["n_vacuoles"] = [n_vac.get(h, 0) for h in df.index]
    df["parasite_area_px"] = [para_area.get(h, 0.0) for h in df.index]
    df["host_area_px_total"] = [total_area.get(h, np.nan) for h in df.index]
    df["on_border"] = [on_border.get(h, False) for h in df.index]
    excl_area = {
        int(h): float((excluded_mask & (host_labels == h)).sum()) for h in host_ids
    }
    df["excluded_area_px"] = [excl_area.get(h, 0.0) for h in df.index]

    df.index.name = "host_id"
    return df


def segment_host_cells(
    image: np.ndarray,
    channel_index: int = 1,
    clip_percentile: float = 99.0,
    diameter: float | None = None,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
    min_area_px: float = 0.0,
    max_area_px: float = 1e9,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Stage H1: segment host cells with cpSAM on a bright-clipped channel.

    Whole-cell segmentation is cpSAM's native task; the only special handling
    is clip_bright(), which stops much-brighter intracellular parasites from
    compressing host contrast during model normalisation.  Only the area gate
    is applied afterwards — spread U2OS fail the PV eccentricity/solidity
    gates, so those are disabled (spec §4 H1).

    Note: cpSAM may propose a bright vacuole inside a NON-expressing cell as a
    small "host" — the min-area gate is what rejects these (spec §4 H1), so
    keep min_area_px well above any vacuole footprint.

    Returns
    -------
    (filtered_labels, raw_labels, stats) — same convention as segment_pvs().
    stats additionally has clip_percentile, clip_value, clip_skipped.
    """
    from ._segment import _filter_by_morphology, _get_model

    if image.ndim == 3:
        chan = image[..., channel_index].astype(np.float32)
    elif image.ndim == 2:
        chan = image.astype(np.float32)
    else:
        raise ValueError(f"Expected 2-D or 3-D array, got shape {image.shape}")

    clipped = clip_bright(chan, clip_percentile)
    if clipped.max() > clipped.min():
        seg_img = clipped
        clip_skipped = False
    else:
        # Clipping flattened the image (or it was already flat) — cpSAM would
        # see no contrast, so fall back to the unclipped channel.
        seg_img = chan
        clip_skipped = True

    model = _get_model()
    masks, _, _ = model.eval(
        seg_img,
        diameter=diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        normalize=True,
    )
    raw_labels = np.asarray(masks).astype(np.int32)

    filtered, stats = _filter_by_morphology(
        raw_labels, min_area_px, max_area_px, 1.0, 0.0
    )
    stats.update(
        {
            "clip_percentile": float(clip_percentile),
            "clip_value": float(seg_img.max()),
            "clip_skipped": clip_skipped,
        }
    )
    return filtered, raw_labels, stats


def measure_vacuoles_in_hosts(
    vac_labels: np.ndarray,
    para_labels: np.ndarray,
    image: np.ndarray,
    vac_to_host: dict[int, int],
    vacuole_map: dict[int, int],
    ch_cptsa: int = 0,
    ch_mcherry: int = 1,
    ch_names: dict[int, str] | None = None,
    pixel_size_um: float | None = None,
) -> pd.DataFrame:
    """
    Measure each parasitophorous vacuole inside an accepted host cell.

    One row per vacuole, indexed by ``vacuole_id``.  Measurements come from
    measure_pvs() run on the Stage-1 vacuole masks, so the ratio is the
    whole-lumen ratio — the correct PV readout, since mCherry fills the entire
    vacuole rather than only the parasite bodies (same reasoning as PV mode).

    Vacuoles with no host assignment in *vac_to_host* are dropped: under the
    population rule only vacuoles inside accepted fluorescent hosts are
    analysed.

    Parameters
    ----------
    vac_labels : np.ndarray (H, W) int32
        Stage-1 vacuole label image.
    para_labels : np.ndarray (H, W) int32
        Per-parasite label image (for the per-vacuole parasite aggregates).
    image : np.ndarray (H, W, C)
    vac_to_host : dict {vacuole_id → host_id}   from assign_to_hosts()
    vacuole_map : dict {parasite_label → vacuole_id}
    ch_cptsa, ch_mcherry, ch_names, pixel_size_um
        As in measure_pvs().

    Returns
    -------
    pd.DataFrame indexed by vacuole_id, with the measure_pvs() columns plus
    ``host_id``, ``parasites_per_vacuole``, ``mean_parasite_ratio`` and
    ``median_parasite_ratio``.  Empty frame when no vacuole is assigned.
    """
    from ._measure import measure_pvs

    if vac_labels is None or vac_labels.max() == 0 or not vac_to_host:
        return pd.DataFrame()

    df = measure_pvs(vac_labels, image, ch_cptsa, ch_mcherry, ch_names, pixel_size_um)
    if df.empty:
        return pd.DataFrame()

    # Keep only vacuoles that belong to an accepted host.
    keep = [v for v in df.index if int(v) in vac_to_host]
    df = df.loc[keep].copy()
    if df.empty:
        return pd.DataFrame()
    df["host_id"] = [vac_to_host[int(v)] for v in df.index]

    # Per-vacuole parasite counts and ratio aggregates.
    para_df = measure_pvs(
        para_labels, image, ch_cptsa, ch_mcherry, ch_names, pixel_size_um
    )
    counts: dict[int, int] = {}
    ratios: dict[int, list[float]] = {}
    for para_label, vac_id in vacuole_map.items():
        vac_id = int(vac_id)
        counts[vac_id] = counts.get(vac_id, 0) + 1
        if not para_df.empty and para_label in para_df.index:
            r = para_df.loc[para_label, "ratio_intden"]
            if not pd.isna(r):
                ratios.setdefault(vac_id, []).append(float(r))

    df["parasites_per_vacuole"] = [counts.get(int(v), 0) for v in df.index]
    df["mean_parasite_ratio"] = [
        float(np.mean(ratios[int(v)])) if int(v) in ratios else np.nan for v in df.index
    ]
    df["median_parasite_ratio"] = [
        float(np.median(ratios[int(v)])) if int(v) in ratios else np.nan
        for v in df.index
    ]
    df.index = df.index.astype(int)
    df.index.name = "vacuole_id"
    return df


def host_vacuole_summary(
    vac_df: pd.DataFrame,
    host_ids: list[int],
) -> pd.DataFrame:
    """
    Aggregate a per-vacuole table (measure_vacuoles_in_hosts) per host cell.

    Returns one row per host in *host_ids* — hosts with no vacuoles get zero
    counts and NaN ratios, so an uninfected host is never confused with a
    missing measurement.  Columns are the per-host vacuole/parasite summary
    folded into hosts.csv.
    """
    cols = [
        "vacuole_area_px_total",
        "vacuole_area_um2_total",
        "mean_vacuole_area_px",
        "mean_parasites_per_vacuole",
        "max_parasites_per_vacuole",
        "mean_vacuole_ratio_intden",
        "median_vacuole_ratio_intden",
        "mean_parasite_ratio",
        "median_parasite_ratio",
    ]
    idx = pd.Index([int(h) for h in host_ids], name="host_id")
    out = pd.DataFrame(index=idx, columns=cols, dtype=float)
    out["vacuole_area_px_total"] = 0.0
    out["vacuole_area_um2_total"] = 0.0
    out["mean_parasites_per_vacuole"] = 0.0
    out["max_parasites_per_vacuole"] = 0

    if vac_df is None or vac_df.empty or "host_id" not in vac_df.columns:
        return out

    for host_id, grp in vac_df.groupby("host_id"):
        h = int(host_id)
        if h not in out.index:
            continue
        out.loc[h, "vacuole_area_px_total"] = float(grp["area_px"].sum())
        if "area_um2" in grp.columns:
            out.loc[h, "vacuole_area_um2_total"] = float(grp["area_um2"].sum())
        out.loc[h, "mean_vacuole_area_px"] = float(grp["area_px"].mean())
        out.loc[h, "mean_parasites_per_vacuole"] = float(
            grp["parasites_per_vacuole"].mean()
        )
        out.loc[h, "max_parasites_per_vacuole"] = int(
            grp["parasites_per_vacuole"].max()
        )
        out.loc[h, "mean_vacuole_ratio_intden"] = float(grp["ratio_intden"].mean())
        out.loc[h, "median_vacuole_ratio_intden"] = float(grp["ratio_intden"].median())
        if grp["mean_parasite_ratio"].notna().any():
            out.loc[h, "mean_parasite_ratio"] = float(grp["mean_parasite_ratio"].mean())
            out.loc[h, "median_parasite_ratio"] = float(
                grp["median_parasite_ratio"].median()
            )
    out["max_parasites_per_vacuole"] = out["max_parasites_per_vacuole"].fillna(0)
    return out
