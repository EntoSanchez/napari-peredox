"""
remeasure_host_run.py - recompute a host-mode batch run from its saved masks.

Reuses the mask TIFFs a previous napari-peredox host batch wrote, so an
analysis change (corrected ratio channels, new vacuole columns, different
exclusion) can be applied without re-running cpSAM segmentation. Equivalent to
the batch widget's "Re-measure: reuse saved masks" checkbox, but scriptable and
reproducible.

Usage
-----
    uv run python scripts/remeasure_host_run.py "<run folder>" \
        --cptsa 2 --mcherry 1 [--images "max projections"] \
        [--treatment Vehicle --cell-line UZS01 --replicate 1]

<run folder> must contain output/masks/ and the image folder (default
"max projections"). Existing output/hosts.csv and output/host_parasites.csv are
moved to output/old_versions/ before the new ones are written.

Caveat: runs predating vacuole-mask saving have no *_host_vac_mask.tif, so the
per-vacuole columns fall back to the parasite mask and describe parasite bodies
rather than PV lumens. The script reports how many positions were affected.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from napari_peredox._batch import _load_cached_masks
from napari_peredox._host import (
    assign_to_hosts,
    assign_vacuoles_to_hosts,
    host_vacuole_summary,
    measure_hosts,
    measure_vacuoles_in_hosts,
)
from napari_peredox._measure import measure_pvs


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
    ap.add_argument("--cptsa", type=int, required=True, help="Peredox channel")
    ap.add_argument("--mcherry", type=int, required=True, help="mCherry channel")
    ap.add_argument("--images", default="max projections")
    ap.add_argument("--dilation-px", type=int, default=3)
    ap.add_argument("--treatment", default="")
    ap.add_argument("--cell-line", default="")
    ap.add_argument("--replicate", type=int, default=1)
    args = ap.parse_args()

    run = Path(args.run_folder)
    mask_dir = run / "output" / "masks"
    img_dir = run / args.images
    out_dir = run / "output"
    if not mask_dir.is_dir():
        raise SystemExit(f"No masks folder: {mask_dir}")

    ch_names = {args.cptsa: "cptsa", args.mcherry: "mcherry"}
    host_rows, para_rows, vac_rows = [], [], []
    n_fallback = 0

    host_masks = sorted(mask_dir.glob("*_host_mask.tif"))
    print(f"{run.name}: {len(host_masks)} position(s) with saved masks")

    for hm in host_masks:
        stem = hm.name.replace("_host_mask.tif", "")
        # Batch names masks "<file_stem>_<safe_pos>_host_mask.tif"; for a
        # TIFF-folder source both halves are the same stem, so the name doubles.
        half = stem[: len(stem) // 2].rstrip("_")
        img_path = img_dir / f"{half}.tif"
        if not img_path.exists():
            cands = list(img_dir.glob(f"{half}*.tif"))
            if not cands:
                print(f"  ! no image for {half} - skipped")
                continue
            img_path = cands[0]

        cached = _load_cached_masks(mask_dir, half, half, host_only=False)
        if cached is None:
            print(f"  ! masks incomplete for {half} - skipped")
            continue
        host_labels, vac_labels, para_labels, vac_map = cached
        if not (mask_dir / f"{half}_{half}_host_vac_mask.tif").exists():
            n_fallback += 1

        image = load_image(img_path)
        px = read_pixel_size(img_path)
        for ch in (args.cptsa, args.mcherry):
            if ch >= image.shape[-1]:
                raise SystemExit(
                    f"{img_path.name} has {image.shape[-1]} channels; "
                    f"channel {ch} is out of range"
                )

        para_to_host, vac_to_host, _dropped = assign_to_hosts(
            para_labels, host_labels, vac_map
        )
        if vac_labels is not None and vac_labels.max() > 0:
            vac_direct, _ = assign_vacuoles_to_hosts(vac_labels, host_labels)
            vac_to_host = {**vac_to_host, **vac_direct}

        hosts_df = measure_hosts(
            host_labels=host_labels,
            para_labels=para_labels,
            image=image,
            para_to_host=para_to_host,
            vac_to_host=vac_to_host,
            dilation_px=args.dilation_px,
            exclude_labels=vac_labels,
            ch_cptsa=args.cptsa,
            ch_mcherry=args.mcherry,
            ch_names=ch_names,
            pixel_size_um=px if px > 0 else None,
        )
        vac_df = measure_vacuoles_in_hosts(
            vac_labels=vac_labels,
            para_labels=para_labels,
            image=image,
            vac_to_host=vac_to_host,
            vacuole_map=vac_map,
            ch_cptsa=args.cptsa,
            ch_mcherry=args.mcherry,
            ch_names=ch_names,
            pixel_size_um=px if px > 0 else None,
        )
        if not hosts_df.empty:
            hosts_df = hosts_df.join(host_vacuole_summary(vac_df, list(hosts_df.index)))

        para_df = measure_pvs(
            labels=para_labels,
            image=image,
            ch_cptsa=args.cptsa,
            ch_mcherry=args.mcherry,
            ch_names=ch_names,
            pixel_size_um=px if px > 0 else None,
        )
        if not para_df.empty:
            para_df["host_id"] = para_df.index.map(para_to_host)
            para_df["vacuole_id"] = para_df.index.map(vac_map)
            para_df = para_df[para_df["host_id"].notna()]

        for df in (hosts_df, para_df, vac_df):
            if df is not None and not df.empty:
                df["file"] = half
                df["position"] = half
                df["treatment"] = args.treatment or run.name
                df["cell_line"] = args.cell_line
                df["replicate"] = args.replicate
        if not hosts_df.empty:
            host_rows.append(hosts_df)
        if para_df is not None and not para_df.empty:
            para_rows.append(para_df)
        if vac_df is not None and not vac_df.empty:
            vac_rows.append(vac_df)
        print(
            f"  {half[-28:]:28s} hosts={len(hosts_df):3d} "
            f"vac={len(vac_df):2d} para={len(para_df):2d}"
        )

    if not host_rows:
        raise SystemExit("Nothing measured.")

    hosts_all = pd.concat(host_rows)
    old_dir = out_dir / "old_versions"
    for name in ("hosts.csv", "host_parasites.csv"):
        p = out_dir / name
        if not p.exists():
            continue
        old_dir.mkdir(parents=True, exist_ok=True)
        try:
            shutil.move(str(p), str(old_dir / name))
        except PermissionError:
            # Open in Excel: copy instead of move so the backup still exists.
            shutil.copy2(str(p), str(old_dir / name))

    def write_csv(df: pd.DataFrame, name: str) -> Path:
        """Write next to the run, falling back if the file is locked."""
        target = out_dir / name
        try:
            df.to_csv(target)
        except PermissionError:
            target = out_dir / name.replace(".csv", "_NEW.csv")
            df.to_csv(target)
            print(f"  ! {name} is open in another program - wrote {target.name}")
        return target

    p = write_csv(hosts_all, "hosts.csv")
    print(f"\nwrote {len(hosts_all)} host rows -> {p}")
    if para_rows:
        paras_all = pd.concat(para_rows)
        p = write_csv(paras_all, "host_parasites.csv")
        print(f"wrote {len(paras_all)} parasite rows -> {p.name}")
    if n_fallback:
        print(
            f"NOTE: {n_fallback} position(s) had no saved vacuole mask - their "
            "per-vacuole columns describe parasite bodies, not PV lumens."
        )
    inf = hosts_all["infected"].astype(bool)
    print(
        f"summary: {len(hosts_all)} hosts, {int(inf.sum())} infected, "
        f"{int(hosts_all['n_vacuoles'].sum())} vacuoles, "
        f"{int(hosts_all['n_parasites'].sum())} parasites"
    )
    print(
        "ratio_intden median - uninfected "
        f"{hosts_all.loc[~inf, 'ratio_intden'].median():.4f} | infected "
        f"{hosts_all.loc[inf, 'ratio_intden'].median():.4f}"
    )


if __name__ == "__main__":
    main()
