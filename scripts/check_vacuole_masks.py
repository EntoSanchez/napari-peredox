"""
check_vacuole_masks.py - QC the vacuole masks against the parasite masks.

The Stage-1 vacuole gate (default 20-2000 um^2) can be larger than a small PV,
so single-parasite vacuoles risk being filtered away. This script reports, per
run, what fraction of detected parasites actually sit inside a detected
vacuole - the number that decides whether the vacuole masks are trustworthy.

Usage
-----
    uv run python scripts/check_vacuole_masks.py "<run folder>" [...]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tifffile
from skimage.measure import regionprops


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_folders", nargs="+")
    args = ap.parse_args()

    for folder in args.run_folders:
        run = Path(folder)
        mask_dir = run / "output" / "masks"
        n_para = n_covered = n_vac = 0
        para_areas: list[float] = []
        vac_areas: list[float] = []
        per_vac_counts: list[int] = []

        for hm in sorted(mask_dir.glob("*_host_mask.tif")):
            stem = hm.name.replace("_host_mask.tif", "")
            vac_p = mask_dir / f"{stem}_host_vac_mask.tif"
            para_p = mask_dir / f"{stem}_host_para_mask.tif"
            if not (vac_p.exists() and para_p.exists()):
                continue
            vac = tifffile.imread(vac_p).astype(np.int32)
            para = tifffile.imread(para_p).astype(np.int32)
            n_vac += len(np.unique(vac)) - 1
            counts: dict[int, int] = {}
            for rp in regionprops(para):
                n_para += 1
                para_areas.append(float(rp.area))
                vals = vac[rp.slice][para[rp.slice] == rp.label]
                winner = int(np.argmax(np.bincount(vals)))
                if winner != 0:
                    n_covered += 1
                    counts[winner] = counts.get(winner, 0) + 1
            for rp in regionprops(vac):
                vac_areas.append(float(rp.area))
                per_vac_counts.append(counts.get(int(rp.label), 0))

        px2 = 0.1083**2
        pct = 100.0 * n_covered / n_para if n_para else 0.0
        print(f"\n{run.name}")
        print(
            f"  parasites: {n_para} | inside a detected vacuole: {n_covered} ({pct:.0f}%)"
        )
        print(f"  vacuoles detected: {n_vac}")
        if para_areas:
            print(
                f"  parasite area um^2: median {np.median(para_areas) * px2:.1f} "
                f"(range {min(para_areas) * px2:.1f}-{max(para_areas) * px2:.1f})"
            )
        if vac_areas:
            print(
                f"  vacuole  area um^2: median {np.median(vac_areas) * px2:.1f} "
                f"(range {min(vac_areas) * px2:.1f}-{max(vac_areas) * px2:.1f})"
            )
        if per_vac_counts:
            empty = sum(1 for c in per_vac_counts if c == 0)
            print(
                f"  parasites per vacuole: median {np.median(per_vac_counts):.0f}, "
                f"max {max(per_vac_counts)}, empty vacuoles {empty}/{len(per_vac_counts)}"
            )


if __name__ == "__main__":
    main()
