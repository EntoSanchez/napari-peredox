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
