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
               VacuoleCurationWidget      accepts object_name: str = "vacuole" —
                                           set to "host cell" for host review
                                           (window title/buttons/status only; no
                                           behavioral change)

_host.py     ← Host-cell segmentation and infection analysis (see "Host analysis
               workflow" below; spec: docs/superpowers/specs/2026-08-25-host-
               analysis-design.md)
               clip_bright()              percentile-clip a channel's non-zero
                                           pixels before cpSAM (Stage H1)
               segment_host_cells()       cpSAM on the bright-clipped host
                                           channel, area gate only (no
                                           eccentricity/solidity gates)
               assign_to_hosts()          majority pixel-overlap parasite/vacuole
                                           → host assignment (Stage H3)
               measure_hosts()            per-host cytosol (host minus dilated
                                           parasites) Peredox ratio + infection
                                           metadata columns (Stage H3)

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

## Host analysis workflow

Second analysis mode, alongside the original PV pipeline. Segments U2OS host
cells expressing cytosolic Peredox, then runs the existing vacuole→parasite
pipeline *inside* those hosts, and reports per-host infection status and
Peredox ratio. Full design: `docs/superpowers/specs/2026-08-25-host-analysis-design.md`.

### Single-image "Host" tab (`_widget.py`, fourth tab)

Stages run in order, each gated behind the previous one's save/completion:

1. **Stage H1 — segment host cells** (`_run_host_stage1` → `_host.segment_host_cells()`
   on a dedicated `_HostWorker` / `_host_thread`, never the PV worker/thread).
   Host-tab widgets: host channel (default mCherry), clip percentile
   (default 99.0), cpSAM diameter, host area gates (µm²), and the parasite
   exclusion buffer (px). Flow/cellprob thresholds are not duplicated in the
   Host tab — Stage H1 reuses the Setup tab's shared cpSAM `flow_threshold` /
   `cellprob_threshold` spinboxes.
   Area gate only — no eccentricity/solidity filter (spread U2OS fail those).
2. **Review hosts** (`_open_host_curation`) — the same accept/reject/redraw
   gallery as PV curation (`VacuoleCurationWidget(object_name="host cell")`).
   Saving appends to `curated_host_features.csv` and retrains
   `curated_host_features.joblib` once both classes have enough examples —
   entirely separate from the PV classifier/CSV. **Stage H2 stays disabled
   until this save happens.**
3. **Stage H2 — segment parasites in hosts** (`_run_host_stage2`) — the image
   is masked to accepted hosts only, then the *existing* Stage 1/2
   vacuole→parasite pipeline runs unchanged (Setup-tab segmentation
   settings), so extracellular parasites and parasites in rejected/dark
   cells never enter the analysis.
4. **Review parasites** (`_open_host_parasite_curation`) — existing parasite
   curation gallery; PV-mode training data/classifier are reused as-is.
5. **Stage H3 — assign, measure, export** (`_run_host_measure` →
   `_host.assign_to_hosts()` + `_host.measure_hosts()`, then
   `_export_host_csv`). Builds the results table and writes the CSV/TIFF
   outputs below.

### Batch mode (`_batch.py`)

`BatchWidget` has an **Analysis mode** dropdown: "Vacuoles / PVs" (existing
behavior, default) or "Host cells". In host mode, `_BatchWorker._process_host_position()`
runs H1 → H2 → H3 automatically for every position; the host classifier (when
trained) filters Stage H1 only — Stage H2 parasite detection in batch host
mode is not classifier-filtered. Host-mask review happens per position via
the same curation gallery used for PV batch review. Output
folder gets `hosts.csv` and `host_parasites.csv` (one row per host / per
parasite across all positions), plus per-position `*_host_mask.tif` /
`*_host_para_mask.tif` label images alongside the existing PV batch outputs.

### Population rule (spec §2)

**All accepted fluorescent hosts are measured, infected or not** — uninfected
hosts in the same image are the internal control group. **Parasites are only
segmented/measured when inside an accepted fluorescent host**; extracellular
parasites and parasites inside non-expressing (dark, non-fluorescent) cells
are excluded by construction (Stage H2 masks the image to accepted hosts
before segmenting).

### Single-image output filenames (`annotations/results/`)

- `<stem>_host_measurements.csv` — one row per accepted host (index
  `host_id`); existing `measure_pvs()` columns computed on the **cytosol**
  (host mask minus parasite pixels dilated by the exclusion buffer, default
  3 px), plus `infected`, `n_parasites`, `n_vacuoles`, `parasite_area_px`,
  `host_area_px_total`, `on_border`, `cytosol_empty`.
- `<stem>_host_parasite_measurements.csv` — one row per accepted parasite,
  existing Stage-2 columns plus `host_id`. Only written when parasites exist.
- `<stem>_host_labels.tif` — accepted host label image.
- `<stem>_host_parasite_labels.tif` — accepted parasite label image (kept
  separate from PV-mode `<stem>_labels.tif`). Only written when parasites exist.

Rejected hosts are zeroed in place rather than removed, and accepted hosts
keep their original cpSAM label IDs — there is no consecutive relabeling —
so `host_id` in the CSVs always matches the label value in
`<stem>_host_labels.tif`; this is a deliberate deviation from the spec's
"relabeled consecutively" line.

### Host classifier files (`annotations/`)

- `curated_host_features.csv` — growing host curation log (same schema
  pattern as `curated_features.csv`: `image_stem`, `label`, `accepted`, +
  FEATURE_COLS). Never mixed with PV annotations.
- `curated_host_features.joblib` — RandomForest trained on the above once
  both classes have enough examples; loaded via
  `load_classifier(annot_dir, filename="curated_host_features.joblib")`.

---

## Tests

Unit tests live in `tests/` (pytest, added as a dev dependency:
`uv add --dev pytest`). Covers `_host.py` (clip_bright edge cases,
majority-overlap assignment, cytosol subtraction/dilation, on_border,
infected/uninfected labeling, NaN propagation), the `object_name`
parameterization of `VacuoleCurationWidget`, host-specific `_io.py` /
`_learning.py` filename plumbing, and a module-import smoke test. cpSAM
inference itself is not unit-tested — the Host tab's segmentation quality is
verified by running it on real images (spec §8 acceptance test).

```bash
uv run pytest tests/ -v
```

32 tests, all passing as of 2026-08-25.

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

## State of the project as of 2026-08-25

- **All core modules written**: `_segment.py`, `_measure.py`, `_learning.py`, `_curation.py`, `_io.py`, `_widget.py`, `_batch.py`, `_host.py`
- **Dependencies installed**: all packages including cellSAM and nd2 installed in `.venv` (Python 3.11)
- **Imports verified**: all modules import cleanly from the venv, including `_host` (`_widget`, `_batch`, `_curation`, `_host`)
- **Host-cell analysis mode added** (`host-analysis` branch, spec
  `docs/superpowers/specs/2026-08-25-host-analysis-design.md`): see "Host
  analysis workflow" above. `tests/` added (pytest, 32 tests passing).
  Not yet run against real U2OS Peredox images end-to-end (spec §8
  acceptance test — cpSAM segmentation quality on real data — is pending
  manual verification).

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
