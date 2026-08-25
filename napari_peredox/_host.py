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
