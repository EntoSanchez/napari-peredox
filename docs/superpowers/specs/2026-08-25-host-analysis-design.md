# Host-Cell Peredox Analysis — Design Spec

**Date:** 2026-08-25
**Status:** Approved by user (design review in chat)
**Scope:** Add host-cell segmentation and analysis to napari-peredox — segment U2OS
host cells expressing cytosolic Peredox with cpSAM, then segment intracellular
parasites (also expressing Peredox) within those hosts, and measure how infection
changes the host-cell Peredox ratio.

---

## 1. Background and goal

The plugin currently runs a two-stage PV pipeline: Stage 1 detects vacuoles
(cpSAM or StarDist), Stage 2 detects individual parasites within accepted
vacuoles, with curation galleries and per-mode active-learning classifiers.

New experimental context:

- **Host cells:** U2OS expressing cytosolic Peredox (cpTSapphire + mCherry).
  Not fully confluent, but cells touch while spread out.
- **Parasites:** also express Peredox, and are **much brighter** than host
  cells in both channels — bright enough to distinguish by eye.
- **Scientific readout:** per-host-cell cpTSapphire/mCherry ratio, compared
  between infected and uninfected hosts. Uninfected cells in the same image
  serve as internal controls.

## 2. Decisions locked during design review

| Decision | Choice |
|---|---|
| Pipeline architecture | **Approach A** — two independent passes + assignment (host cpSAM pass, then existing PV pipeline masked to accepted hosts, then majority-overlap assignment) |
| Host curation | Yes — reuse the accept/reject/redraw gallery **and** a host-specific RandomForest classifier (separate training data from PVs) |
| Border-touching hosts | Keep and measure, add `on_border` flag column |
| Host ratio measurement | Exclude parasite pixels **dilated by a buffer** (default 3 px, configurable) from the host mask before measuring |
| UI placement | New **"Host" tab** in the existing `PeredoxWidget` |
| Batch | Build single-image and batch modes **simultaneously**; batch gets an analysis-mode dropdown |
| Analysis population | **All accepted fluorescent hosts are measured, infected or not** (uninfected hosts are the control group). **Parasites are only segmented/measured when inside an accepted fluorescent host** — extracellular parasites and parasites in non-expressing (dark) cells are excluded. |

## 3. Outputs

Per analyzed image, written to the annotations dir via `_io.py` conventions:

### 3.1 `<stem>_host_measurements.csv` — one row per accepted host cell

- All existing `measure_pvs()` columns (centroids, shape descriptors,
  per-channel intensity stats, `ratio_intden` / `ratio_mean` /
  `ratio_median`), measured on the **cytosol mask** = host mask minus
  parasite pixels dilated by `parasite_dilation_px` (default 3).
- `host_id` (index) — label ID in the saved host label image.
- `infected` (bool) — host contains ≥ 1 accepted parasite.
- `n_parasites` (int), `n_vacuoles` (int) — accepted objects assigned to this host.
- `parasite_area_px` (float) — total accepted parasite pixels inside this host.
- `host_area_px_total` (float) — full host footprint area (before subtraction),
  so the excluded fraction is computable.
- `on_border` (bool) — host mask touches any image edge.
- `cytosol_empty` (bool) — True when subtraction leaves zero pixels; ratio
  columns are NaN in that case, row is kept.

### 3.2 `<stem>_host_parasite_measurements.csv` — one row per parasite

Same columns as the existing Stage-2 parasite measurements (including
per-vacuole joins), plus `host_id` (int) linking each parasite to its host.

### 3.3 Label images

- `<stem>_host_labels.tif` — accepted host label image.
- `<stem>_host_parasite_labels.tif` — accepted parasite label image
  (host-mode run, kept separate from PV-mode outputs).

## 4. Pipeline

### Stage H1 — host segmentation

1. Take the host segmentation channel (UI dropdown, default = mCherry channel).
2. **Clip bright pixels**: intensities above the `clip_percentile`-th
   percentile of non-zero pixels (default 99.0, configurable) are clipped to
   that percentile value. Rationale: cpSAM normalizes input intensity; very
   bright parasites would otherwise compress host cells into the bottom of
   the dynamic range.
3. Run cpSAM (the module-global cached model from `_segment._get_model()` —
   shared with the PV pipeline, loaded once per session). Parameters exposed:
   `diameter` (default None = auto), `flow_threshold`, `cellprob_threshold`.
4. Morphology filter: **area gate only** by default
   (`min_area_um2` / `max_area_um2`, converted to px² using pixel size).
   No eccentricity/solidity gates — spread U2OS are irregular and the PV
   gates would wrongly reject them.
   Known failure mode: a bright parasite/vacuole inside a **non-expressing**
   cell can appear to cpSAM as a small object on dark background and be
   proposed as a "host". The min-area gate rejects these (a vacuole is far
   below any plausible U2OS footprint); host curation and the host
   classifier are the backstop. This enforces the population rule in §2 —
   such parasites must never enter the analysis.
5. Optional host classifier filter (see §6) once trained.

### Host review

The Stage-H1 labels open in the single-object curation gallery
(accept / reject / redraw polygon). Saving:

- keeps accepted hosts only (relabeled consecutively),
- appends decisions + features to `curated_host_features.csv`,
- retrains/updates `host_classifier.joblib` when both classes have enough
  examples (same thresholds as the PV classifier).

### Stage H2 — parasites within hosts

1. Build the union mask of **accepted** hosts; zero the image outside it.
   Extracellular parasites and parasites in rejected hosts are excluded by
   construction.
2. Run the **existing** vacuole→parasite pipeline on the masked image,
   unchanged: `segment_pvs()` (with the existing intensity-threshold options —
   percentile/Otsu recommended defaults here since parasites are much
   brighter) → existing morphology gates → optional StarDist stage 2 via
   `segment_parasites_in_vacuoles()`.
3. Existing parasite curation gallery reviews the result. PV-mode training
   data and models are reused as-is (parasite appearance is the same task).

### Stage H3 — assignment and measurement

1. **Assignment:** each accepted parasite is assigned to the host label with
   **majority pixel overlap** with the parasite mask. Ties broken by the
   larger overlap count first, then lower host ID (deterministic). A parasite
   whose majority pixel is background (possible only at mask edges after
   curation redraws) is dropped from host statistics and logged.
   Vacuole assignment: a vacuole belongs to the host holding the majority of
   its member-parasite pixels.
2. **Host measurement:** build the cytosol label image (host labels with
   dilated parasite pixels removed), run existing `measure_pvs()` on it, then
   join the §3.1 metadata columns.
3. **Parasite measurement:** existing Stage-2 measurement path, plus the
   `host_id` column.

## 5. Code structure

### New module: `napari_peredox/_host.py`

| Function | Contract |
|---|---|
| `clip_bright(channel, percentile) -> np.ndarray` | Clip a 2-D float array at the given percentile of its non-zero pixels; returns a copy. Percentile ≥ 100 or an all-zero image → unmodified copy. |
| `segment_host_cells(image, channel_index, clip_percentile, diameter, flow_threshold, cellprob_threshold, min_area_px, max_area_px) -> (filtered, raw, stats)` | Steps 1–4 of Stage H1. Same return convention as `segment_pvs()`. |
| `assign_to_hosts(para_labels, host_labels, vacuole_map) -> (para_to_host: dict[int,int], vac_to_host: dict[int,int], dropped: list[int])` | Majority-overlap assignment per §4/H3. |
| `measure_hosts(host_labels, para_labels, image, para_to_host, dilation_px, ch_cptsa, ch_mcherry, ch_names, pixel_size_um) -> pd.DataFrame` | Cytosol construction + `measure_pvs()` delegation + metadata columns per §3.1 (including `on_border`, `infected`, counts, `cytosol_empty`). |

Everything except the cpSAM call inside `segment_host_cells()` is pure
array/DataFrame logic and unit-testable with synthetic inputs.

### Modified modules

- **`_widget.py`** — new `_build_host_tab()` (fourth tab, "Host") and a
  `_HostWorker(QObject)` following the `_VacuoleWorker` pattern. Tab
  contents: host channel dropdown (default mCherry), clip percentile
  spinbox, cpSAM diameter / flow / cellprob spinboxes, host area gates in
  µm², parasite-exclusion dilation spinbox (px, default 3), and the button
  sequence *Segment host cells → Review hosts → Segment parasites in hosts →
  Review parasites → Measure & export*. Stage H2/H3 buttons stay disabled
  until the previous stage completes. Log panel shared with the Segment tab.
  Host-mode state lives in new attributes (`_host_labels`,
  `_host_para_labels`, `_host_measurements`, …) and never mutates PV-mode
  state.
- **`_curation.py`** — `VacuoleCurationWidget` gains an `object_name: str`
  parameter (default `"vacuole"`) used in window title, buttons, and status
  text. No behavioral change.
- **`_learning.py`** — no logic change; host mode calls
  `train_classifier()` / `load_classifier()` with host-specific paths.
  `load_classifier()` gets an optional filename parameter
  (default `curated_features.joblib` for backward compatibility).
- **`_io.py`** — save/append helpers accept the host-mode filenames of §3;
  `append_curated_annotations()` gets a target-CSV parameter (default
  unchanged).
- **`_batch.py`** — `BatchWidget` gains an **analysis mode** dropdown
  ("Vacuoles/PVs" — current behavior, default — vs "Host cells").
  In host mode `_BatchWorker.run()` calls the same `_host.py` functions per
  position: Stage H1 → Stage H2 (auto, classifier-filtered if available) →
  Stage H3, appending per-host and per-parasite rows to two output CSVs
  (`<output_stem>_hosts.csv`, `<output_stem>_host_parasites.csv`).
  The existing manual-review flow is extended so host masks are reviewable
  per position, same as PV curation is today. Host-tab defaults
  (clip percentile, dilation, area gates) are duplicated in the batch
  channels tab.

### File layout in the annotations dir

```
annotations/
  curated_features.csv          (existing, PV)
  curated_features.joblib       (existing, PV)
  curated_host_features.csv     (new)
  host_classifier.joblib        (new)
  results/
    <stem>_host_measurements.csv
    <stem>_host_parasite_measurements.csv
    <stem>_host_labels.tif
    <stem>_host_parasite_labels.tif
```

## 6. Host classifier (active learning)

Identical machinery to the PV classifier: `extract_features()` over host
labels, decisions appended to `curated_host_features.csv`,
RandomForest trained by `train_classifier()` once ≥ 10 examples with both
classes exist, saved as `host_classifier.joblib`, applied in Stage H1 when the
"use classifier" checkbox is on. Host and PV training data never mix.

## 7. Error handling

- Zero hosts found → log message, Stage H2 button stays disabled.
- Zero parasites found → hosts are all `infected = False`; measurement still
  runs (this is a valid uninfected-control image).
- `cytosol_empty` hosts → NaN ratios, row kept, count logged.
- Dropped parasites (no host majority) → logged with label IDs.
- Worker exceptions → traceback to the log panel via the existing
  `error` signal pattern.
- Clip percentile so aggressive it flattens the image (clipped max == min) →
  fall back to unclipped channel, log a warning (mirrors the existing
  threshold-fallback pattern in `segment_pvs()`).

## 8. Testing

- Add `pytest` as a dev dependency (`uv add --dev pytest`); tests in `tests/`.
- Unit tests with synthetic label/intensity arrays for: `clip_bright`
  (including all-zero and ≥100-percentile edge cases), majority-overlap
  assignment (clean containment, straddling two hosts, tie, background
  majority), cytosol subtraction with dilation (including full-coverage →
  `cytosol_empty`), `on_border` flagging, infected/uninfected labeling, and
  NaN propagation into ratio columns.
- cpSAM inference is not unit-tested; acceptance test is running the Host tab
  on real U2OS Peredox images (user validation).
- ruff clean after every edit (project policy).

## 9. Out of scope (explicitly)

- 3-D / Z-resolved analysis (batch continues to max-project).
- Time-lapse tracking of hosts across frames.
- Retraining StarDist on host cells (cpSAM only for hosts).
- Any change to the existing PV-mode outputs or file formats.
