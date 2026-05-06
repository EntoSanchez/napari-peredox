# napari-peredox

A [napari](https://napari.org) plugin for segmenting parasitophorous vacuoles (PVs) and measuring **Peredox** ratiometric fluorescence in *T. gondii* infection experiments.

**Peredox** is a genetically encoded NADH sensor built from two fluorescent proteins:
- **cpTSapphire** — emission shifts with the NADH/NAD⁺ ratio
- **mCherry** — ratiometric reference, redox-insensitive

The primary readout is `cpTSapphire / mCherry` (integrated density ratio), which reports the relative NADH level inside each PV. Higher ratio = more reduced environment.

---

## Features

| Widget | What it does |
|--------|-------------|
| **Peredox: Segment & Measure PVs** | Single-image workflow: segment PVs with Cellpose-SAM, measure fluorescence, curate results |
| **Peredox: Batch Process ND2 Files** | High-throughput: process every XY stage position in one or more ND2 Z-stack files |
| **Peredox: Curate PV Segments** | Standalone curation panel (normally launched from the two widgets above) |

Additional capabilities:
- **Active learning**: every curation session trains a RandomForest classifier that automatically filters false positives in future sessions
- **Vacuole grouping**: nearby parasites are clustered into vacuoles; one representative parasite is selected per vacuole for reporting
- **Measurement export**: area, shape descriptors, per-channel intensity statistics, cpTSapphire/mCherry ratio — all exported to CSV

---

## Requirements

| Requirement | Notes |
|-------------|-------|
| Python 3.11 | Earlier versions not tested |
| NVIDIA GPU (recommended) | Cellpose-SAM runs on CPU but is ~10× slower; CUDA 12.4 for the default install |
| napari ≥ 0.5 | Installed automatically |
| Cellpose ≥ 4.1 | Uses the `cpsam` (Cellpose-SAM) model |

On first run, Cellpose downloads the `cpsam` model weights (~600 MB) from HuggingFace to `~/.cellpose/models/`. Internet access is required for this one-time download.

---

## Installation

### Option A — uv (recommended, includes CUDA PyTorch)

[uv](https://docs.astral.sh/uv/) manages the virtual environment and resolves CUDA-enabled PyTorch automatically.

```bash
# Install uv if you don't have it
# Windows (PowerShell):
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
# macOS/Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone https://github.com/lourido-lab/napari-peredox.git
cd napari-peredox
uv sync                     # creates .venv and installs all dependencies
```

To launch napari:

```bash
# Windows (PowerShell):
.\.venv\Scripts\napari.exe

# Windows (Git Bash):
source .venv/Scripts/activate
napari

# macOS / Linux:
source .venv/bin/activate
napari
```

### Option B — pip (bring your own PyTorch)

If you already have a working PyTorch + CUDA environment (e.g. a conda env):

```bash
git clone https://github.com/lourido-lab/napari-peredox.git
cd napari-peredox
pip install .
```

If you need to install PyTorch with CUDA support first:

```bash
# CUDA 12.4:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install .
```

For CPU-only (slower segmentation):

```bash
pip install .   # torch from PyPI is CPU-only by default
```

---

## Quick start

### Single-image workflow

1. Open napari and load your image (`File → Open…` or drag and drop).
   - Accepts ND2, TIFF, or any format napari supports.
   - Multi-channel images are expected as **(C, H, W)** — napari's default order.

2. Open the plugin: **Plugins → Peredox: Segment & Measure PVs**.

3. In the **Setup** panel:
   - Select your image layer.
   - Set **cpTSapphire channel** and **mCherry channel** indices (0-based).
   - Set **Pixel size (µm)** — required for the µm² area filter. If left at 0, the plugin will prompt you when you click Run.

4. Click **▶ Segment PVs**. The log shows progress; segmentation runs in a background thread so napari stays responsive.

5. After segmentation, two label layers appear:
   - `<stem>_parasite_candidates` — all Cellpose-SAM detections (dimmed)
   - `<stem>_parasites` — after morphology filtering (area, eccentricity, solidity)

6. Click **🔍 Open Curation Panel** to step through each PV thumbnail and accept/reject. When done, click **💾 Save & retrain classifier**.

7. Click **💾 Export measurements CSV** to save a CSV and label TIFF to `annotations/results/`.

### Batch workflow

Open **Plugins → Peredox: Batch Process ND2 Files**. The batch widget supports two image sources — choose at the top of the panel.

#### Source A — ND2 Z-stack files (default)

Use this when starting from raw acquisition files.

1. Add ND2 files using **Add files…** or **Add folder…** (recursively finds all `.nd2` files in a folder).
2. Fill in **Treatment**, **Cell line**, and **Replicate** metadata.
3. Configure channels and segmentation parameters.
4. Choose an **Output folder** and click **▶ Run batch**.

For each XY stage position in each ND2 file the plugin:
- Max-projects the Z-stack → saves as `out_folder/mips/<stem>_<pos>_MIP.tif`
- Segments parasites → saves mask as `out_folder/masks/<stem>_<pos>_mask.tif`
- Measures fluorescence and appends to the in-memory results table

#### Source B — TIFF folder (Max IPs)

Use this when you already have max-intensity projection TIFFs — for example, MIPs exported from NIS-Elements, Fiji, or a previous batch run. Switch the **Source type** dropdown to **TIFF folder (Max IPs)**, then choose the folder containing your `.tif` / `.tiff` files. Each file is treated as one position; the file stem becomes the position name in the results CSV.

Accepted TIFF layouts:
- **(H, W)** — single channel, gains a dummy channel axis automatically
- **(C, H, W)** — ImageJ/Fiji convention (first axis ≤ 16 assumed to be channels)
- **(H, W, C)** — already in pipeline format

Pixel size is read from the TIFF metadata if present (OME-TIFF `PhysicalSizeX` or standard `XResolution` tag); otherwise enter it manually in the **Pixel size** field.

#### After either batch run

Use the **Manual review** panel to open any position in the curation gallery, accept/reject individual parasites, then click **Save accepted results CSV** to write `out_folder/results.csv`. Positions you have not reviewed default to keeping all detected parasites.

---

## Segmentation parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| **cpTSapphire channel** | 0 | Channel index for the NADH sensor |
| **mCherry channel** | 1 | Channel index for the reference |
| **Segment on channel** | 0 | Channel Cellpose-SAM sees (or tick "Max composite") |
| **Diameter (px)** | 0 (auto) | Expected parasite diameter; 0 = Cellpose auto-estimates |
| **Flow threshold** | 0.4 | Higher = more permissive (finds more objects) |
| **Cell prob threshold** | 0.0 | Lower = more permissive; try −1 or −2 for dim PVs |
| **Intensity threshold** | none | Pre-masks background before segmentation; otsu/percentile/manual |
| **Area filter (µm²)** | 5 – 25 | Discard segments outside this physical area range |
| **Max eccentricity** | 0.95 | Discard elongated objects (1 = line, 0 = circle) |
| **Min solidity** | 0.60 | Discard fragmented objects (1 = perfectly convex) |
| **Group into vacuoles** | on | Clusters nearby parasites; reports one per vacuole |
| **Dilation (px)** | 5 | Expansion radius used when grouping parasites |
| **Select parasite by** | largest | Which parasite represents the vacuole: `largest`, `highest_ratio`, `median_ratio` |

**Recommended settings for Peredox experiments:**

- **Intensity threshold → percentile** (50–75): set the threshold channel to cpTSapphire (usually ch 0). This zeros out dim autofluorescent background before Cellpose-SAM runs, which substantially reduces false-positive detections in cells with no parasites. Start at the 50th percentile (median of non-zero pixels) and increase if too many background objects are still being segmented.
- **Select parasite by → median_ratio**: within each vacuole, picks the parasite whose cpTSapphire/mCherry ratio is closest to the group median rather than the largest or brightest parasite. This is more robust to outliers caused by parasites at the edge of the focal plane.

---

## Output files

### `annotations/results/<stem>_measurements.csv`

One row per PV (after vacuole grouping). Key columns:

| Column | Description |
|--------|-------------|
| `area_px` | Segment area in pixels |
| `area_um2` | Physical area (µm²), if pixel size is set |
| `circularity` | 4π·A / P² — 1 = perfect circle |
| `aspect_ratio` | Major / minor axis length |
| `mean_cptsa`, `intden_cptsa` | Mean and integrated density for cpTSapphire |
| `mean_mcherry`, `intden_mcherry` | Mean and integrated density for mCherry |
| `ratio_cptsa_mcherry` | Primary Peredox readout (`intden_cptsa / intden_mcherry`) |
| `parasites_per_vacuole` | Number of parasites in the containing vacuole |

Additional shape descriptors and per-channel statistics (median, std, min, max, mode, skewness, kurtosis) are included for each channel.

### `out_folder/results.csv` (batch mode)

Same columns plus experiment metadata: `file`, `position_index`, `position_name`, `treatment`, `cell_line`, `replicate`.

---

## Active learning classifier

Every time you curate a session and click **Save & retrain classifier**, the plugin:

1. Appends your accept/reject decisions to `annotations/curated_features.csv`
2. Trains a `RandomForestClassifier` on morphological and intensity features
3. Saves the model to `annotations/curated_features.joblib`

From the next segmentation onwards, the classifier is loaded automatically. Enable it with the **Apply classifier filter** checkbox (disabled by default until you have enough data).

Training requires **≥ 10 annotated examples with at least one accept and one reject**. The classifier status panel shows current counts.

The `annotations/` directory is excluded from version control (`.gitignore`) — it contains your lab's data and trained model, which should be kept locally or stored separately.

---

## Project structure

```
napari_peredox/
├── _widget.py      Main dock widget (single-image workflow)
├── _batch.py       Batch widget + ND2/TIFF reader
├── _segment.py     Cellpose-SAM wrapper + morphology filter
├── _measure.py     Fluorescence measurements (regionprops + IntDen)
├── _curation.py    Accept/reject UI with thumbnails
├── _learning.py    Feature extraction + RandomForest training
├── _io.py          CSV and TIFF persistence
└── napari.yaml     napari plugin manifest
```

---

## Development

```bash
# Clone and set up dev environment
git clone https://github.com/lourido-lab/napari-peredox.git
cd napari-peredox
uv sync

# Lint and format after editing
uv run ruff check --fix napari_peredox/ && uv run ruff format napari_peredox/

# Verify imports
uv run python -c "from napari_peredox._widget import PeredoxWidget; print('OK')"
```

---

## Citation

If you use this plugin in published work, please cite the Peredox paper:

> Hung, Y.P., Albeck, J.G., Tantama, M., & Yellen, G. (2011). Imaging cytosolic NADH-NAD+ redox state with a genetically encoded fluorescent biosensor. *Cell Metabolism*, 14(4), 545–554.
