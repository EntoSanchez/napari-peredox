"""
backfill_vacuole_masks.py - add missing *_host_vac_mask.tif to an old host run.

Host-mode runs from before vacuole masks were saved have host and parasite
masks on disk but no Stage-1 vacuole masks, so a re-measure can only fall back
to the parasite bodies and cannot report true PV-lumen counts or ratios.

This script runs ONLY the Stage-1 vacuole detection (cpSAM on the host-masked
image, the same call `_BatchWorker._process_host_position` makes) and writes
the missing mask next to the existing ones. Host and parasite masks are left
untouched, so nothing about the earlier segmentation or its curation changes -
afterwards `remeasure_host_run.py` produces genuine vacuole measurements.

Usage
-----
    uv run python scripts/backfill_vacuole_masks.py "<run folder>" \
        --vac-seg-ch 1 [--vac-min-area-um2 20 --vac-max-area-um2 2000] [--force]

GPU note: one cpSAM call per position (~20-40 s on a free RTX 3060). Close
other GPU-heavy apps first; VRAM contention makes it far slower.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import tifffile

from napari_peredox._batch import save_mask_tiff
from napari_peredox._segment import preload_model, segment_pvs


def read_pixel_size(path: Path) -> float:
    with tifffile.TiffFile(path) as tf:
        tag = tf.pages[0].tags.get("XResolution")
        if tag is not None and tag.value[0]:
            return float(tag.value[1]) / float(tag.value[0])
    return 0.0


def load_image(path: Path) -> np.ndarray:
    arr = tifffile.imread(path).astype(np.float32)
    if arr.ndim == 3:
        arr = np.moveaxis(arr, int(np.argmin(arr.shape)), -1)
    elif arr.ndim == 2:
        arr = arr[..., np.newaxis]
    return arr


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_folder")
    ap.add_argument("--vac-seg-ch", type=int, default=1)
    ap.add_argument("--images", default="max projections")
    ap.add_argument("--vac-min-area-um2", type=float, default=20.0)
    ap.add_argument("--vac-max-area-um2", type=float, default=2000.0)
    ap.add_argument("--max-eccentricity", type=float, default=0.95)
    ap.add_argument("--min-solidity", type=float, default=0.60)
    ap.add_argument("--force", action="store_true", help="redo existing masks")
    ap.add_argument(
        "--from-parasites",
        action="store_true",
        help=(
            "Build vacuoles by grouping the existing parasite masks (dilate, "
            "connected components, fill holes) instead of running cpSAM. Use "
            "when Stage-1 detection latches onto host nuclei rather than PVs, "
            "which is what happens on U2OS+Peredox mCherry images. No GPU."
        ),
    )
    ap.add_argument("--dilation-px", type=int, default=5)
    args = ap.parse_args()

    run = Path(args.run_folder)
    mask_dir = run / "output" / "masks"
    img_dir = run / args.images
    if not mask_dir.is_dir():
        raise SystemExit(f"No masks folder: {mask_dir}")

    host_masks = sorted(mask_dir.glob("*_host_mask.tif"))
    todo = []
    for hm in host_masks:
        stem = hm.name.replace("_host_mask.tif", "")
        out = mask_dir / f"{stem}_host_vac_mask.tif"
        if out.exists() and not args.force:
            continue
        todo.append((hm, stem, out))

    print(
        f"{run.name}: {len(todo)} of {len(host_masks)} position(s) need vacuole masks"
    )
    if not todo:
        return

    if args.from_parasites:
        from scipy.ndimage import binary_fill_holes
        from skimage.measure import label as sk_label
        from skimage.morphology import dilation as morph_dilation
        from skimage.morphology import disk

        for i, (hm, stem, out) in enumerate(todo, 1):
            para_p = mask_dir / f"{stem}_host_para_mask.tif"
            if not para_p.exists():
                print(
                    f"  [{i}/{len(todo)}] no parasite mask for {stem[-26:]} - skipped"
                )
                continue
            para = tifffile.imread(para_p).astype(np.int32)
            if para.max() == 0:
                save_mask_tiff(np.zeros_like(para), out)
                print(f"  [{i}/{len(todo)}] {stem[-26:]:26s} no parasites - empty mask")
                continue
            # Parasites sharing a vacuole are adjacent: dilate, take connected
            # components, fill holes.  Same idea as _segment.group_by_vacuole()
            # and _io._build_vacuole_mask(), which PV mode already relies on.
            selem = disk(args.dilation_px)
            grown = morph_dilation(para > 0, selem)
            groups = sk_label(binary_fill_holes(grown), connectivity=2)
            vac = np.zeros_like(para)
            n = 0
            for gid in np.unique(groups):
                if gid == 0:
                    continue
                region = groups == gid
                if not (para[region] > 0).any():
                    continue  # dilation artefact with no parasite inside
                n += 1
                vac[region] = n
            save_mask_tiff(vac, out)
            print(
                f"  [{i}/{len(todo)}] {stem[-26:]:26s} "
                f"{len(np.unique(para)) - 1:2d} parasites -> {n:2d} vacuoles",
                flush=True,
            )
        print("done (no GPU used)")
        return

    print(preload_model())

    t_start = time.time()
    for i, (hm, stem, out) in enumerate(todo, 1):
        half = stem[: len(stem) // 2].rstrip("_")
        img_path = img_dir / f"{half}.tif"
        if not img_path.exists():
            cands = list(img_dir.glob(f"{half}*.tif"))
            if not cands:
                print(f"  [{i}/{len(todo)}] no image for {half} - skipped")
                continue
            img_path = cands[0]

        host_labels = tifffile.imread(hm).astype(np.int32)
        if host_labels.max() == 0:
            save_mask_tiff(np.zeros_like(host_labels), out)
            print(f"  [{i}/{len(todo)}] {half[-26:]:26s} no hosts - empty mask written")
            continue

        image = load_image(img_path)
        px = read_pixel_size(img_path)
        if px > 0:
            vac_min_px = args.vac_min_area_um2 / (px**2)
            vac_max_px = args.vac_max_area_um2 / (px**2)
        else:
            vac_min_px, vac_max_px = 0.0, 1e9

        # Same call the batch worker makes for Stage H2's vacuole stage.
        masked = image * (host_labels > 0)[..., np.newaxis].astype(image.dtype)
        t0 = time.time()
        vac_labels, _raw, stats = segment_pvs(
            image=masked,
            channel_index=args.vac_seg_ch,
            use_composite=False,
            min_area_px=vac_min_px,
            max_area_px=vac_max_px,
            max_eccentricity=args.max_eccentricity,
            min_solidity=args.min_solidity,
            diameter=None,
            flow_threshold=0.4,
            cellprob_threshold=0.0,
            threshold_method="none",
        )
        save_mask_tiff(vac_labels, out)
        print(
            f"  [{i}/{len(todo)}] {half[-26:]:26s} "
            f"{stats['total_raw']:3d} raw -> {stats['kept']:3d} vacuoles "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )

    print(f"done in {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
