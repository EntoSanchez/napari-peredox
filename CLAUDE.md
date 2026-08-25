# napari-peredox — Claude Code Project Guide

## Python Environment Policy
- **Always use the project venv**: `source .venv/Scripts/activate` (Git Bash on Windows)
- Verify correct Python with `which python` — must point to `.venv/Scripts/python`
- **Package manager: uv** — use `uv add <pkg>`, never bare `pip install`
- Dev dependencies: `uv add --dev <pkg>`
- **Linter/formatter: ruff** — run `uv run ruff check --fix napari_peredox/ && uv run ruff format napari_peredox/` after editing any Python file

---

## Project Overview

napari-peredox is a napari plugin for:
1. **Segmenting parasitophorous vacuoles (PVs)** in 2D fluorescence microscopy using cellSAM
2. **Measuring ratiometric Peredox fluorescence**: cpTSapphire integrated density / mCherry integrated density (proxy for intracellular NADH/NAD⁺ ratio)
3. **Curating segments**: accept/reject UI to screen false positives
4. **Learning over time**: a RandomForest classifier trained on curated features pre-filters cellSAM output in future sessions

---

## Architecture

```
_widget.py   ← Main dock widget (napari entry point)
               ├── QThread _SegmentWorker  (non-blocking cellSAM call)
               ├── calls _segment.py       → label array
               ├── calls _measure.py       → measurements DataFrame
               ├── calls _learning.py      → features DataFrame
               ├── launches _curation.py   → CurationWidget (floating window)
               └── calls _io.py            → CSV / TIFF persistence

_segment.py  ← cellSAM wrapper
               segment_pvs()              runs cellSAM, filters by area
               apply_classifier_filter()  removes false positives using sklearn model

_measure.py  ← Fluorescence measurement
               measure_pvs()              regionprops + integrated density per channel
               summary_stats()            mean/std across all accepted PVs

_learning.py ← Active learning
               extract_features()         morphology + intensity features per segment
               train_classifier()         RandomForestClassifier on curated CSV
               load_classifier()          load .joblib from annotations dir
               classifier_stats()         count accept/reject examples

_batch.py    ← Bulk ND2 processing
               read_nd2_positions()       reads all stage points from one ND2 file
                                           returns list of (name, (H,W,C) array)
                                           Z-stacks are max-projected automatically
               _extract_position_names()  pulls stage point names from ND2 metadata
               _BatchWorker               QObject run in a QThread; emits progress signals
               BatchWidget                dock widget: file list, metadata fields,
                                           channel config, output CSV, progress bar
               make_batch_widget()        napari entry point

_curation.py ← Curation UI
               CurationWidget             scrollable gallery: thumbnail + accept/reject
               make_curation_widget()     napari entry point (placeholder)

_io.py       ← Persistence
               save_measurements()        → annotations/results/<stem>_measurements.csv
               save_labels()              → annotations/results/<stem>_labels.tif
               append_curated_annotations() → annotations/curated_features.csv (append)
               load_measurements()        ← read back measurements
               load_labels()              ← read back label image
```

---

## Key Data Structures

### Image format
The plugin expects napari `Image` layers. Internally, all images are converted to
`(H, W, C) float32` numpy arrays in `_widget.py._get_image_array()`.
napari stores multi-channel images as `(C, H, W)` — the widget transposes this.

### Label array
`np.ndarray (H, W) int32`. Background = 0, PV 1 = 1, PV 2 = 2, etc.
Produced by `segment_pvs()` in `_segment.py`.

### Measurements DataFrame
`pd.DataFrame` indexed by `label` (int). Key columns:
- `centroid_y`, `centroid_x` — pixel coordinates
- `area_px` — pixel count
- `area_um2` — physical area (only if pixel_size_um > 0)
- `mean_<ch_name>`, `intden_<ch_name>` — per channel
- `ratio_cptsa_mcherry` — primary output

### Features DataFrame
Same index as measurements. Columns from `_learning.FEATURE_COLS`.
Both DataFrames are available as `self._features` and `self._measurements` on the widget.

### Annotation CSV (`annotations/curated_features.csv`)
Columns: `image_stem`, `label`, `accepted`, + all FEATURE_COLS.
`accepted` = 1 (true PV) or 0 (false positive). `-1` / NaN = skipped, excluded from training.
Rows are **appended** each session. Re-curating the same image+label overwrites the old row.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| napari | Viewer framework |
| cellSAM | PV segmentation (from GitHub) |
| scikit-image | regionprops, label, find_boundaries |
| scikit-learn | RandomForestClassifier, Pipeline, StandardScaler |
| joblib | Classifier serialisation |
| pandas | Measurement tables, annotation CSV |
| numpy | Array operations |
| tifffile | Reading/writing TIFF images |
| qtpy | Qt bindings (PyQt5 or PySide2 via napari) |
| torch/torchvision | Required by cellSAM |

---

## Workflow for adding new features

1. Read the affected module(s) before editing
2. Keep measurement logic in `_measure.py`, UI logic in `_widget.py`
3. Feature columns for the classifier are defined in `_learning.FEATURE_COLS` — add new features there and they will automatically be included in training
4. Run ruff after every edit
5. Test with `uv run python -c "from napari_peredox.<module> import <name>; print('OK')"`

---

## Known limitations / future work

- **3D support**: Single-image widget is 2D only (takes middle slice for Z-stacks). Batch mode handles Z-stacks via max-intensity projection automatically.
- **Batch MIP saving**: each position's MIP is saved as a (C,H,W) float32 TIFF immediately after reading, before segmentation — crash-safe and Fiji-compatible (imagej=True).
- **Classifier cold start**: for the first ~10 images the classifier has no data; all cellSAM output goes to the curation panel unfiltered. This is by design.
- **cellSAM model download**: first run requires internet access to download weights to `~/.deepcell/`. Subsequent runs use the cache.
- **GPU**: cellSAM will use CUDA if available. On CPU it is slower but functional.

---

## State of the project as of 2026-04-30

- **All core modules written**: `_segment.py`, `_measure.py`, `_learning.py`, `_curation.py`, `_io.py`, `_widget.py`, `_batch.py`
- **Dependencies installed**: all packages including cellSAM and nd2 installed in `.venv` (Python 3.11)
- **Imports verified**: all modules import cleanly from the venv

### Recent changes (2026-04-30 session)

**`_segment.py`**
- Replaced Feret diameter filter with physical area filter in `_filter_by_morphology()`: uses `rp.area` vs `[min_area_px, max_area_px]`; stat key is `rejected_area`
- `segment_pvs()` returns `tuple[np.ndarray, np.ndarray, dict]` with keys `total_raw`, `rejected_area`, `rejected_eccentricity`, `rejected_solidity`, `kept`
- `group_by_vacuole()`: replaced deprecated `binary_dilation` with `dilation` from `skimage.morphology`
- Added `preload_model() -> str`: loads model in main thread before worker starts, logs device (e.g. "Cellpose-SAM loaded on GPU (cuda:0) — 609 MB used")
- `_get_model()` now calls `torch.cuda.init()` + `torch.zeros(1, device="cuda")` to force CUDA context creation before Cellpose loads

**`_widget.py`**
- Area filter UI changed from Feret spinboxes to `_min_area_um2` / `_max_area_um2` (µm², defaults 5.0/25.0)
- Pixel size prompt before segmentation: `QInputDialog.getDouble()` if `pixel_size == 0`
- Added classifier checkbox `_use_classifier` (default unchecked)
- Calls `preload_model()` from main thread before starting the worker thread

**`_batch.py`**
- Same area filter UI changes (µm²)
- TIFF file stem fix: uses `tp.stem` (actual TIFF filename) not `tiff_folder.name`
- Manual review workflow: `_result_df` holds raw output; `_curation_decisions` maps `(file, position_name)` → `{label: 0/1}`; `_save_accepted_results()` filters rejected labels; auto-save removed
- `_curation_win` stored as instance attribute to prevent GC
- Added `_use_classifier` checkbox (default unchecked)
- Calls `preload_model()` from main thread before `self._thread.start()`

**`_measure.py`**
- Added `from scipy import stats as scipy_stats`
- New shape columns per region: `area_fraction`, `perimeter`, `circularity`, `aspect_ratio`, `roundness`, `eccentricity`, `solidity`, `extent`, `major_axis_length`, `minor_axis_length`, `orientation`
- Uses `rp.axis_major_length` / `rp.axis_minor_length` (not deprecated versions)
- New per-channel columns: `median_<ch>`, `std_<ch>`, `min_<ch>`, `max_<ch>`, `mode_<ch>`, `skewness_<ch>`, `kurtosis_<ch>`
- `select_one_per_vacuole()`: fixed to preserve label index (was dropping label IDs)

**`_curation.py`**
- Fixed last-parasite not annotatable: `_accept`/`_reject` now always call `_refresh()` regardless of position
- `_next()` no-op on last item but still calls `_refresh()` so status updates

### Status
- **Not yet tested on real data**: plugin has not been run against actual Peredox microscopy images
- **No real training data yet**: `annotations/` directory is empty; classifier cannot train until ≥10 curated examples exist
- **GPU verified**: `preload_model()` called in main thread; CUDA context initialized before worker starts

---

## Before compaction — save these notes

When context is approaching compaction, save the following to this file:
- Which files were recently modified and why
- Any bugs discovered and how they were fixed
- Any design decisions that deviated from the original plan
- Status of testing on real data
- Current count of annotations in curated_features.csv
