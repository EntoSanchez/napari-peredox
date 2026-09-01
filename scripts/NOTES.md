# scripts/ — notes

## host_param_sweep.py

Offline cpSAM host-segmentation tuning: runs `_host.segment_host_cells` over a
clip-percentile × diameter grid on one MIP TIFF and writes a boundary-overlay
panel PNG next to the input. No area gate is applied so the panel shows raw
cpSAM output; pick area gates from the reported median object areas.

### First real-data run (2026-08-26)

Image: `Imaging/20260821 range testing uzs01/Vehicle/1n/max projections/`
`60xSil_561mCherry_405peredox_Vehicle_UZS01_range__pos002_MIP.tif`
(U2OS + cytosolic Peredox, vehicle, 60× silicone, 0.1083 µm/px, 2304²,
ch0 = cpTSapphire/405, ch1 = mCherry/561; dim signal — camera offset ~100,
cytosol p50 ≈ 145 counts on ch1).

Findings:
- **diameter auto is unusable on this data**: 168–282 objects, median 1 µm²
  (image shatters into speckle). This was the cause of the "only segmenting
  nuclei" observation — never leave host diameter on auto for these images.
- Any explicit diameter (150/300/450 px) produces whole-cell segmentations:
  10–19 objects, median 630–880 µm² (spread U2OS scale).
- Clip percentile 99 vs 90 vs 85 barely changes detection on uninfected
  images (the clip lever matters when bright parasites are present).

Recommended starting batch settings for this dataset:
- Segmentation channel: 1 (mCherry)
- Host diameter: **300 px** (≈ 32 µm at 0.1083 µm/px)
- Clip percentile: 99 (default)
- Host area gate: min **350 µm²** (default 200 lets nucleus-scale junk pass;
  U2OS nucleus ≈ 150–250 µm²), max 10000 µm²
- Ratio channels for this scope layout: cpTSapphire = 0, mCherry = 1

## remeasure_host_run.py

Recomputes a host-mode batch run from its saved mask TIFFs — the scriptable
form of the batch widget's "Re-measure: reuse saved masks" checkbox. Use when
an analysis change (corrected channels, new columns, different exclusion)
should be applied without paying for cpSAM segmentation again.

```
uv run python scripts/remeasure_host_run.py "<run folder>" --cptsa 2 --mcherry 1 \
    --treatment Vehicle --cell-line UZS01
```

Existing `output/hosts.csv` / `host_parasites.csv` are moved to
`output/old_versions/` first (copied instead, if open in Excel); a locked
target is written as `*_NEW.csv` rather than lost.

### Run 2026-09-01 — 20260824 UZS01 Infection / KO

Re-measured `KO/Vehicle` and `KO/1_nM` with **cptsa (Peredox) = ch2,
mCherry = ch1**. The original batch had run with cptsa = ch0, which is the
640 IMC1-Halo channel — every ratio in those CSVs was Halo/mCherry, not
Peredox/mCherry. Old files preserved in each `output/old_versions/`.

| Arm | Hosts | Infected | Uninf. median ratio | Inf. median ratio | Mann-Whitney |
|---|---|---|---|---|---|
| Vehicle | 237 | 19 | 1.3475 | 1.2853 | p = 0.073 |
| 1_nM | 185 | 22 | 1.3084 | 1.3175 | p = 0.897 |

**Vacuole masks (2026-09-01, second pass).** Both runs predated vacuole-mask
saving. Two approaches were tried:

1. `backfill_vacuole_masks.py` (cpSAM Stage-1 on host-masked mCherry, the
   batch worker's own call) — **rejected**. QC showed only 51-62 % of
   parasites landed inside a detected vacuole and 62-66 % of "vacuoles" were
   empty; the overlay showed cpSAM had latched onto **host nuclei**, which are
   the roundest high-contrast objects once the image is masked to hosts. The
   20 um^2 Stage-1 floor also sits above a single PV (~12 um^2 here).
2. `backfill_vacuole_masks.py --from-parasites` — **used**. Vacuoles are the
   connected components of the parasite masks dilated 5 px and hole-filled
   (`_segment.group_by_vacuole` / `_io._build_vacuole_mask` logic, which PV
   mode already relies on). QC: 100 % of parasites inside a vacuole, zero
   empty vacuoles, 1-5 parasites per vacuole. The mask is a tight envelope
   around the parasite rosette, so `mean_vacuole_ratio_intden` is a
   rosette-plus-margin ratio, not a full-lumen ratio.

`check_vacuole_masks.py` runs that QC; always run it after a backfill.

### Final numbers (curation-preserving re-measure, ch2/ch1)

| Arm | Hosts | Infected | Vacuoles | Parasites | Uninf. median | Inf. median | Mann-Whitney |
|---|---|---|---|---|---|---|---|
| Vehicle | 229 | 18 | 25 | 46 | 1.3480 | 1.2868 | p = 0.046 |
| 1_nM | 180 | 22 | 37 | 53 | 1.3051 | 1.3171 | p = 0.775 |

Row counts match the original curated CSVs exactly (229 / 180) because
`--curated-from` restored the review decisions the saved masks had lost.
`n_vacuoles` now differs from `n_parasites` for 52 % of infected hosts (it was
identical for 100 % under the parasite-mask fallback).

**Open issue:** the QC overlay
(`scripts/` render, 2026-09-01) shows bright PV rosettes in some cells with no
parasite mask on them. Either those cells were not accepted as hosts (correct
by the population rule) or Stage-2 missed them. Worth checking before treating
the infection rate (~8-12 % of hosts) as final.
