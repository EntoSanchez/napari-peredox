# napari-peredox — Project Notes

## What this project is

Peredox is a genetically encoded, ratiometric NADH sensor composed of two fluorescent proteins:
- **cpTSapphire** — circularly permuted T-Sapphire, a cyan/green FP whose emission changes with the NADH/NAD⁺ ratio
- **mCherry** — red FP used as a ratiometric reference (insensitive to redox state)

The ratio **cpTSapphire / mCherry** (integrated density or mean intensity) reports the relative NADH level inside each parasitophorous vacuole (PV). Higher ratio = more reduced environment.

This plugin:
1. Uses **cellSAM** to segment individual PVs in 2D fluorescence images
2. Measures fluorescence intensities inside each segment
3. Lets the user curate (accept/reject) segments to remove false positives
4. **Learns from curation over time**: accepted/rejected examples train a lightweight
   RandomForest classifier that pre-filters segments in future sessions

---

## Measurement definitions

| Term | Formula | Fiji equivalent |
|------|---------|-----------------|
| `mean_<ch>` | Σ pixels / N pixels | Mean |
| `intden_<ch>` | Σ pixel values in mask | RawIntDen |
| `ratio_cptsa_mcherry` | `intden_cptsa / intden_mcherry` | — |

**Note**: `intden_cptsa / intden_mcherry` equals `mean_cptsa / mean_mcherry` because area cancels, but `intden` is preserved for compatibility with Fiji workflows.

---

## Channel assignment

Typical Peredox experiment image:
- Channel 0: cpTSapphire (excitation ~405 nm, emission ~512 nm)
- Channel 1: mCherry (excitation ~587 nm, emission ~610 nm)
- Channel 2+ (optional): DAPI, brightfield, etc.

Adjust channel spinboxes in the Setup panel to match your acquisition.

---

## Segmentation approach

cellSAM (Segment Anything for Cells) is a SAM-based model fine-tuned for cell/organelle segmentation. It is run on a single channel or the channel-wise maximum projection (controlled by the "Use composite" checkbox in the UI).

**Good choices for the segmentation channel:**
- cpTSapphire: usually gives good contrast for PVs
- Max composite: useful if neither channel alone shows all PVs clearly

After cellSAM, segments are filtered by area (configurable min/max).

---

## Active learning / classifier

After each curation session, accepted (true PV) and rejected (false positive) segments are saved to `annotations/curated_features.csv`. The features stored per segment are:

| Feature | Description |
|---------|-------------|
| `area_px` | Segment area in pixels |
| `eccentricity` | 0=circle, 1=line |
| `solidity` | area / convex_hull_area |
| `extent` | area / bounding_box_area |
| `perimeter` | Segment perimeter |
| `mean_intensity_seg_ch` | Mean intensity on segmentation channel |
| `std_intensity_seg_ch` | Std dev of intensity on segmentation channel |
| `mean_cptsa` | Mean cpTSapphire intensity |
| `mean_mcherry` | Mean mCherry intensity |
| `intden_cptsa` | Integrated density cpTSapphire |
| `intden_mcherry` | Integrated density mCherry |
| `ratio_cptsa_mcherry` | cpTSapphire / mCherry ratio |

A `RandomForestClassifier` (100 trees, max_depth=8, balanced class weights) is trained on these features. From the **second session onwards**, the classifier is loaded automatically and false positives are filtered before the curation panel opens, reducing manual review load.

The classifier requires **≥ 10 annotated examples with at least one of each class** before it will train. The status panel in the widget shows current counts.

---

## Batch processing (ND2 files)

Use **Plugins → Peredox: Batch Process ND2 Files** to analyse entire experiments.

The batch widget accepts one or more ND2 files and iterates over every XY stage
position in each file.  For each position it runs the full pipeline
(segmentation → classifier filter → measurement) and appends results to a single
CSV.  You choose an **output folder** (not just a CSV path); all outputs go there.

**Metadata fields** (written as columns in the CSV):

| Field | Example | Notes |
|-------|---------|-------|
| Treatment | "DMSO", "compound_X" | Free text |
| Cell line | "HFF", "RH" | Free text |
| Replicate # | 1, 2, 3 | Integer spinner |

**Output folder layout**:

```
<out_folder>/
├── results.csv          one row per PV, all conditions combined
└── mips/
    ├── file1_pos000_MIP.tif
    ├── file1_pos001_MIP.tif
    └── ...
```

**Output CSV columns** (one row per PV):

```
file, position_index, position_name,
treatment, cell_line, replicate,
pv_label, centroid_y, centroid_x,
area_px, [area_um2],
mean_cptsa, intden_cptsa,
mean_mcherry, intden_mcherry,
ratio_cptsa_mcherry
```

**MIP TIFFs**: saved as (C, H, W) float32, `imagej=True` for Fiji compatibility.
Each position's MIP is saved immediately after reading — before segmentation —
so partial runs leave behind usable images even if processing is interrupted.

**Appending behaviour**: if `results.csv` already exists, new rows are appended
rather than overwriting. This lets you process multiple experimental conditions
into a single file by running batch with different metadata for each set of ND2 files.

**ND2 dimension handling**: the `nd2` library reads multi-position acquisitions
where each XY point is one 'P' frame. Z-stacks are max-projected along Z before
segmentation. Position names from the microscope stage log are preserved; if
names are unavailable they fall back to `pos000`, `pos001`, etc.

---

## File layout

```
napari-peredox/
├── napari_peredox/          Python package (the plugin)
│   ├── __init__.py
│   ├── napari.yaml          napari entry points
│   ├── _widget.py           Main dock widget (single image)
│   ├── _batch.py            Batch widget + ND2 reader
│   ├── _segment.py          cellSAM wrapper
│   ├── _measure.py          Fluorescence measurements
│   ├── _curation.py         Accept/reject UI
│   ├── _learning.py         Feature extraction + classifier training
│   └── _io.py               Persistence (CSV, TIFF)
├── annotations/             Created automatically on first run
│   ├── curated_features.csv Grows each session
│   ├── curated_features.joblib  Trained classifier
│   └── results/             Per-image measurement CSVs + label TIFFs
├── pyproject.toml           uv project config + dependencies
├── uv.lock                  Pinned dependency versions
├── NOTES.md                 This file
└── CLAUDE.md                Developer notes for Claude Code
```

---

## Environment

- **Python**: 3.11
- **Package manager**: uv (`uv add <pkg>` to add dependencies, never pip)
- **Activate venv**: `source .venv/Scripts/activate` (Git Bash on Windows)
- **cellSAM**: installed from `git+https://github.com/vanvalenlab/cellSAM.git`
- **Linter**: ruff (`uv run ruff check --fix && uv run ruff format`)

---

## Running the plugin

```powershell
# In PowerShell from d:/Lourido Lab/napari-peredox/
.\.venv\Scripts\Activate.ps1
napari
# Single image:  Plugins → Peredox: Segment & Measure PVs
# Batch ND2:     Plugins → Peredox: Batch Process ND2 Files
```

---

## Key decisions & rationale

- **cellSAM over StarDist/Cellpose**: cellSAM requires no training data and handles irregular PV shapes better than StarDist (which is tuned for round nuclei). Cellpose also works but cellSAM is simpler to set up for this use case.
- **Post-hoc classifier over fine-tuning SAM**: SAM fine-tuning requires a GPU and many annotated masks (>50). A feature-based RF classifier needs ~10 examples, trains in <1 second, and is transparent (feature importances are inspectable).
- **Integrated density for ratio**: Matches the "RawIntDen" column used in Fiji ImageJ, making comparisons to legacy measurements straightforward.
- **Background thread for segmentation**: cellSAM (especially model download + inference) can take 10–60 seconds. Running in a QThread keeps the napari UI responsive.
