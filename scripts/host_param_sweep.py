"""host_param_sweep.py — offline cpSAM host-segmentation parameter sweep.

Runs napari_peredox._host.segment_host_cells across a grid of clip
percentiles x diameters on one saved MIP TIFF and writes a boundary-overlay
comparison panel (PNG) next to the input image.  Use it to pick the Host
diameter / clip percentile / area gates before a batch run, without
clicking through napari.

Usage (from the napari-peredox project root):
    uv run python scripts/host_param_sweep.py "path/to/image_MIP.tif" \
        [--channel 1] [--diameters 0 150 300 450] [--clips 99 90 85]

Notes:
- diameter 0 means cpSAM auto-estimate.
- No area gate is applied (min_area_px=0) so the panel shows exactly what
  cpSAM produces; pick the area gate from the reported median object areas.
- Pixel size is read from the TIFF's ImageJ XResolution tag when present.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tifffile


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("image", help="Path to a multi-channel MIP TIFF")
    ap.add_argument("--channel", type=int, default=1, help="Segmentation channel")
    ap.add_argument(
        "--diameters",
        type=float,
        nargs="*",
        default=[0, 150, 300, 450],
        help="cpSAM diameters in px (0 = auto)",
    )
    ap.add_argument(
        "--clips",
        type=float,
        nargs="*",
        default=[99.0, 90.0, 85.0],
        help="clip_bright percentiles",
    )
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage.segmentation import find_boundaries

    from napari_peredox._host import clip_bright, segment_host_cells

    arr = tifffile.imread(args.image).astype(np.float32)
    if arr.ndim == 3:
        caxis = int(np.argmin(arr.shape))
        arr = np.moveaxis(arr, caxis, -1)
    elif arr.ndim == 2:
        arr = arr[..., np.newaxis]
    else:
        raise SystemExit(f"Unsupported image shape {arr.shape}")

    px_um = None
    with tifffile.TiffFile(args.image) as tf:
        xr = tf.pages[0].tags.get("XResolution")
        if xr is not None and xr.value[0]:
            px_um = xr.value[1] / xr.value[0]

    chan = arr[..., args.channel]
    rows, cols = len(args.clips), len(args.diameters)
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 4.6 * rows))
    axes = np.atleast_2d(axes)

    for i, clip in enumerate(args.clips):
        for j, diam in enumerate(args.diameters):
            d = None if not diam else float(diam)
            labels, _raw, _stats = segment_host_cells(
                arr,
                channel_index=args.channel,
                clip_percentile=clip,
                diameter=d,
                min_area_px=0.0,
            )
            n = int(len(np.unique(labels)) - 1)

            # Display: the clipped channel, contrast-stretched, with red
            # object boundaries.
            disp = clip_bright(chan, clip)
            lo, hi = np.percentile(disp, (1.0, 99.5))
            disp = np.clip((disp - lo) / max(hi - lo, 1e-6), 0, 1)
            rgb = np.dstack([disp, disp, disp])
            rgb[find_boundaries(labels, mode="outer")] = [1.0, 0.2, 0.2]

            areas = np.bincount(labels.ravel())[1:]
            areas = areas[areas > 0]
            med_px = float(np.median(areas)) if areas.size else 0.0
            area_txt = (
                f", med {med_px * px_um**2:.0f} µm²" if (px_um and med_px) else ""
            )
            title = (
                f"clip p{clip:g}, diam {'auto' if d is None else int(d)} "
                f"→ {n} objects{area_txt}"
            )
            ax = axes[i, j]
            ax.imshow(rgb, interpolation="nearest")
            ax.set_title(title, fontsize=9)
            ax.axis("off")
            # Windows consoles may be cp1252 — print an ASCII-safe variant
            print(title.encode("ascii", "replace").decode(), flush=True)

    name = Path(args.image).name
    sup = f"{name}  (ch{args.channel}"
    sup += f", {px_um:.4f} µm/px)" if px_um else ")"
    fig.suptitle(sup, fontsize=11)
    out = Path(args.image).with_name(
        Path(args.image).stem + f"_host_sweep_ch{args.channel}.png"
    )
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print("saved:", out)


if __name__ == "__main__":
    main()
