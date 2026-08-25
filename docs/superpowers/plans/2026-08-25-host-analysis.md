# Host-Cell Peredox Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a host-cell analysis mode to napari-peredox: segment U2OS host cells expressing cytosolic Peredox with cpSAM, segment the much-brighter parasites inside accepted hosts, and measure per-host Peredox ratio (parasite pixels excluded) with infected/uninfected classification.

**Architecture:** New pure-logic module `_host.py` (clipping, host segmentation, majority-overlap assignment, host measurement) delegates to existing `measure_pvs()` / `_get_model()` / `_filter_by_morphology()`. A fourth "Host" tab in `PeredoxWidget` drives Stage H1 (hosts) → curation → Stage H2 (existing vacuole→parasite pipeline on the host-masked image) → curation → Stage H3 (assign + measure + export). `BatchWidget` gets an analysis-mode dropdown whose host path calls the same `_host.py` functions per position.

**Tech Stack:** Python 3.11, uv, numpy/pandas/scikit-image, Cellpose-SAM (cpsam) via torch, Qt via qtpy, pytest (new dev dep), ruff.

**Spec:** `docs/superpowers/specs/2026-08-25-host-analysis-design.md`

## Global Constraints

- Work in `d:/Lourido Lab/napari-peredox/`; always use the project venv via `uv run …` (never base Python).
- After every Python edit run: `uv run ruff check --fix napari_peredox/ tests/ && uv run ruff format napari_peredox/ tests/`
- Run tests with: `uv run pytest tests/ -v`
- Python target 3.11; line length 88 (ruff config already in pyproject.toml).
- **No change to existing PV-mode outputs, file formats, or behavior.** All host-mode files use new names (spec §3).
- Host analysis population rule (spec §2): every accepted fluorescent host is measured, infected or not; parasites are only analyzed inside accepted fluorescent hosts.
- Host classifier files: `curated_host_features.csv` + `curated_host_features.joblib` (the `.joblib` name follows `train_classifier()`'s existing convention of saving next to the CSV).
- Commit after every task (git repo exists; `main` branch).

**Existing interfaces you will reuse (do not modify):**

```python
# _segment.py
_get_model()                          # cached cpsam CellposeModel; .eval(img, diameter=, flow_threshold=, cellprob_threshold=, normalize=True) -> (masks, _, _)
_filter_by_morphology(labels, min_area_px, max_area_px, max_eccentricity, min_solidity) -> (filtered, stats)
segment_pvs(image, channel_index=, use_composite=, model_path=, min_area_px=, max_area_px=, max_eccentricity=, min_solidity=, diameter=, flow_threshold=, cellprob_threshold=, threshold_method=, threshold_channel=, threshold_value=, threshold_percentile=, watershed_split=, watershed_min_distance=) -> (filtered, raw, stats)
segment_parasites_in_vacuoles(image, vac_labels, seg_channel=, model=, min_area_px=, max_area_px=, max_eccentricity=, min_solidity=, pad=, progress_cb=) -> (para_labels, vacuole_map)
apply_classifier_filter(labels, features, classifier) -> filtered_labels
preload_model() -> str

# _measure.py
measure_pvs(labels, image, ch_cptsa, ch_mcherry, ch_names=None, pixel_size_um=None) -> pd.DataFrame  # indexed by label; empty labels -> empty df
select_one_per_vacuole(df, vacuole_map, method="largest") -> pd.DataFrame
measure_vacuoles(labels, image, vacuole_map, ch_cptsa, ch_mcherry, ch_names=None, pixel_size_um=None) -> pd.DataFrame
add_estimated_parasites(df) -> pd.DataFrame

# _learning.py
extract_features(labels, image, seg_channel=0, ch_cptsa=0, ch_mcherry=1, ch_names=None) -> pd.DataFrame
train_classifier(csv_path) -> clf | None    # saves csv_path.with_suffix(".joblib")
classifier_stats(csv_path) -> dict

# _io.py
save_measurements(df, image_stem, annotations_dir) -> Path   # results/<stem>_measurements.csv
save_labels(labels, image_stem, annotations_dir) -> Path     # results/<stem>_labels.tif
# NOTE: calling these with image_stem=f"{stem}_host" yields exactly the spec §3
# filenames (<stem>_host_measurements.csv etc.) — no _io changes needed for results.

# _stardist.py
load_stardist_model(model_dir, mode="vacuoles"|"parasites")
```

---

### Task 1: pytest infrastructure

**Files:**
- Modify: `pyproject.toml` (via `uv add`)
- Test: `tests/test_imports.py`

**Interfaces:**
- Produces: a working `uv run pytest tests/ -v` invocation used by every later task.

- [ ] **Step 1: Add pytest as a dev dependency**

Run: `uv add --dev pytest`
Expected: `pyproject.toml` `[dependency-groups] dev` gains `pytest`, `uv.lock` updated.

- [ ] **Step 2: Write a smoke test**

Create `tests/test_imports.py`:

```python
"""Smoke tests: every plugin module must import cleanly from the venv."""

import importlib

import pytest

MODULES = [
    "napari_peredox._segment",
    "napari_peredox._measure",
    "napari_peredox._learning",
    "napari_peredox._io",
]


@pytest.mark.parametrize("mod", MODULES)
def test_module_imports(mod):
    importlib.import_module(mod)
```

(`_widget`, `_curation`, `_batch` are excluded: importing them pulls in Qt, which needs a display; they are covered by the import checks inside their own tasks.)

- [ ] **Step 3: Run the tests**

Run: `uv run pytest tests/ -v`
Expected: 4 passed.

- [ ] **Step 4: Ruff, then commit**

```bash
uv run ruff check --fix napari_peredox/ tests/ && uv run ruff format napari_peredox/ tests/
git add tests/ pyproject.toml uv.lock
git commit -m "test: add pytest dev dependency and import smoke tests"
```

---

### Task 2: `_host.py` — `clip_bright()`

**Files:**
- Create: `napari_peredox/_host.py`
- Test: `tests/test_host_clip.py`

**Interfaces:**
- Produces: `clip_bright(channel: np.ndarray, percentile: float = 99.0) -> np.ndarray` — float32 copy of `channel` with values above the `percentile`-th percentile of its **non-zero** pixels clipped to that percentile value. `percentile >= 100` or an all-zero image returns an unmodified float32 copy.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_host_clip.py`:

```python
import numpy as np

from napari_peredox._host import clip_bright


def test_clips_bright_pixels_to_percentile_of_nonzero():
    # 99 dim pixels of value 10, one very bright pixel of 1000
    chan = np.full((10, 10), 10.0, dtype=np.float32)
    chan[0, 0] = 1000.0
    out = clip_bright(chan, percentile=99.0)
    cutoff = np.percentile(chan[chan > 0], 99.0)
    assert out.max() == np.float32(cutoff)
    assert out[5, 5] == 10.0  # dim pixels untouched


def test_zero_pixels_excluded_from_percentile():
    # Mostly zeros; percentile must come from the non-zero values only
    chan = np.zeros((10, 10), dtype=np.float32)
    chan[0, :5] = 100.0
    out = clip_bright(chan, percentile=50.0)
    # 50th percentile of the five 100-valued pixels is 100 -> nothing clipped
    assert out.max() == 100.0


def test_percentile_100_is_noop():
    chan = np.array([[1.0, 5000.0]], dtype=np.float32)
    out = clip_bright(chan, percentile=100.0)
    np.testing.assert_array_equal(out, chan)


def test_all_zero_image_returned_unchanged():
    chan = np.zeros((4, 4), dtype=np.float32)
    out = clip_bright(chan, percentile=99.0)
    np.testing.assert_array_equal(out, chan)


def test_returns_copy_not_view():
    chan = np.full((4, 4), 7.0, dtype=np.float32)
    out = clip_bright(chan, percentile=99.0)
    out[0, 0] = -1
    assert chan[0, 0] == 7.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_host_clip.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'napari_peredox._host'`

- [ ] **Step 3: Create `napari_peredox/_host.py` with the implementation**

```python
"""
_host.py — Host-cell segmentation and infection analysis

Purpose
-------
Implements the host-cell analysis mode (spec:
docs/superpowers/specs/2026-08-25-host-analysis-design.md):

  Stage H1  segment_host_cells() — cpSAM on the host channel with bright
            pixels clipped first, so very bright intracellular parasites do
            not compress the host cells' dynamic range during cpSAM's
            intensity normalisation.
  Stage H3  assign_to_hosts()   — majority-overlap parasite→host assignment.
            measure_hosts()     — per-host Peredox ratio on the cytosol
            (host mask minus dilated parasite pixels) plus infection
            metadata columns.

Stage H2 (parasite detection inside accepted hosts) reuses the existing PV
pipeline in _segment.py on a host-masked image and needs no code here.

Population rule (spec §2): every accepted fluorescent host is measured,
infected or not; parasites are only analysed inside accepted fluorescent
hosts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def clip_bright(channel: np.ndarray, percentile: float = 99.0) -> np.ndarray:
    """
    Clip a single-channel image at the given percentile of its non-zero pixels.

    Used before cpSAM host segmentation: parasites are far brighter than the
    host cytosol, and without clipping they dominate cpSAM's intensity
    normalisation, flattening host contrast.

    Parameters
    ----------
    channel : np.ndarray (H, W)
        Single-channel image (any numeric dtype).
    percentile : float
        Percentile (0–100) of non-zero pixel values used as the clip ceiling.
        Values >= 100 disable clipping.

    Returns
    -------
    np.ndarray (H, W) float32
        Clipped copy. All-zero input or percentile >= 100 returns an
        unmodified float32 copy.
    """
    out = np.asarray(channel, dtype=np.float32).copy()
    if percentile >= 100.0:
        return out
    nonzero = out[out > 0]
    if nonzero.size == 0:
        return out
    cutoff = np.float32(np.percentile(nonzero, percentile))
    np.clip(out, None, cutoff, out=out)
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_host_clip.py -v`
Expected: 5 passed.

- [ ] **Step 5: Ruff, then commit**

```bash
uv run ruff check --fix napari_peredox/ tests/ && uv run ruff format napari_peredox/ tests/
git add napari_peredox/_host.py tests/test_host_clip.py
git commit -m "feat: add _host.clip_bright for pre-cpSAM bright-pixel clipping"
```

---

### Task 3: `_host.py` — `assign_to_hosts()`

**Files:**
- Modify: `napari_peredox/_host.py`
- Test: `tests/test_host_assign.py`

**Interfaces:**
- Produces: `assign_to_hosts(para_labels, host_labels, vacuole_map=None) -> tuple[dict[int, int], dict[int, int], list[int]]` returning `(para_to_host, vac_to_host, dropped)`. Majority pixel overlap decides the host; ties resolve to the lowest host ID; a parasite whose plurality pixel value is background (0) goes to `dropped`. `vac_to_host` maps each vacuole ID from `vacuole_map` (parasite label → vacuole ID) to the host holding the majority of its member-parasite pixels; empty dict when `vacuole_map` is None.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_host_assign.py`:

```python
import numpy as np

from napari_peredox._host import assign_to_hosts


def _canvas():
    """20x20 image: host 1 = left half, host 2 = right half, 2-col background gap."""
    hosts = np.zeros((20, 20), dtype=np.int32)
    hosts[:, 0:9] = 1
    hosts[:, 11:20] = 2
    return hosts


def test_clean_containment():
    hosts = _canvas()
    paras = np.zeros_like(hosts)
    paras[2:5, 2:5] = 1  # fully inside host 1
    p2h, v2h, dropped = assign_to_hosts(paras, hosts)
    assert p2h == {1: 1}
    assert dropped == []


def test_straddling_parasite_goes_to_majority_host():
    hosts = _canvas()
    paras = np.zeros_like(hosts)
    paras[5, 6:14] = 1  # cols 6-8 in host1 (3 px), 11-13 in host2 (3 px), 9-10 bg (2 px)
    paras[6, 6:9] = 1  # 3 more px in host 1 -> host 1 majority
    p2h, _, dropped = assign_to_hosts(paras, hosts)
    assert p2h == {1: 1}
    assert dropped == []


def test_tie_resolves_to_lowest_host_id():
    hosts = _canvas()
    paras = np.zeros_like(hosts)
    paras[5, 7:9] = 1   # 2 px in host 1
    paras[5, 11:13] = 1  # 2 px in host 2
    p2h, _, _ = assign_to_hosts(paras, hosts)
    assert p2h == {1: 1}


def test_background_majority_is_dropped():
    hosts = _canvas()
    paras = np.zeros_like(hosts)
    paras[5, 8:12] = 1  # 1 px host1, 2 px background (cols 9,10), 1 px host2
    p2h, _, dropped = assign_to_hosts(paras, hosts)
    assert p2h == {}
    assert dropped == [1]


def test_vacuole_assignment_follows_member_pixel_majority():
    hosts = _canvas()
    paras = np.zeros_like(hosts)
    paras[2:4, 2:4] = 1   # 4 px, host 1
    paras[2:4, 5:7] = 2   # 4 px, host 1
    paras[10:16, 12:18] = 3  # 36 px, host 2
    vacuole_map = {1: 10, 2: 10, 3: 20}
    p2h, v2h, _ = assign_to_hosts(paras, hosts, vacuole_map)
    assert p2h == {1: 1, 2: 1, 3: 2}
    assert v2h == {10: 1, 20: 2}


def test_no_vacuole_map_returns_empty_vac_dict():
    hosts = _canvas()
    paras = np.zeros_like(hosts)
    paras[2:5, 2:5] = 1
    _, v2h, _ = assign_to_hosts(paras, hosts)
    assert v2h == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_host_assign.py -v`
Expected: FAIL — `ImportError: cannot import name 'assign_to_hosts'`

- [ ] **Step 3: Append the implementation to `_host.py`**

```python
def assign_to_hosts(
    para_labels: np.ndarray,
    host_labels: np.ndarray,
    vacuole_map: dict[int, int] | None = None,
) -> tuple[dict[int, int], dict[int, int], list[int]]:
    """
    Assign each parasite (and vacuole) to the host cell it overlaps most.

    Majority pixel overlap decides ownership.  np.bincount + argmax makes the
    tie-break deterministic: equal counts resolve to the lowest index, i.e.
    background (0) first, then the lowest host ID.  A parasite whose plurality
    value is background is *dropped* — after Stage H2 masking this can only
    happen at mask edges following curation redraws.

    Parameters
    ----------
    para_labels : np.ndarray (H, W) int32
        Parasite label image (0 = background).
    host_labels : np.ndarray (H, W) int32
        Accepted host label image (0 = background).
    vacuole_map : dict {parasite_label → vacuole_id}, optional
        Stage 2 grouping.  When given, each vacuole is assigned to the host
        holding the majority of its member-parasite pixels.

    Returns
    -------
    para_to_host : dict {parasite_label → host_id}
    vac_to_host : dict {vacuole_id → host_id}   (empty if vacuole_map is None)
    dropped : list[int]
        Parasite labels whose plurality pixel was background.
    """
    from collections import defaultdict

    from skimage.measure import regionprops

    para_to_host: dict[int, int] = {}
    dropped: list[int] = []
    # Per-vacuole pixel counts per host, accumulated across member parasites
    vac_counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))

    for rp in regionprops(para_labels):
        mask = para_labels[rp.slice] == rp.label
        host_vals = host_labels[rp.slice][mask]
        counts = np.bincount(host_vals)
        winner = int(np.argmax(counts))
        if winner == 0:
            dropped.append(int(rp.label))
        else:
            para_to_host[int(rp.label)] = winner

        if vacuole_map is not None:
            vac_id = vacuole_map.get(int(rp.label))
            if vac_id is not None:
                for hid in np.nonzero(counts)[0]:
                    if hid != 0:
                        vac_counts[int(vac_id)][int(hid)] += int(counts[hid])

    vac_to_host: dict[int, int] = {}
    for vac_id, cmap in vac_counts.items():
        # max count wins; ties resolve to the lowest host ID
        best_host = min(cmap, key=lambda hid: (-cmap[hid], hid))
        vac_to_host[vac_id] = best_host

    return para_to_host, vac_to_host, dropped
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_host_assign.py -v`
Expected: 6 passed.

- [ ] **Step 5: Ruff, then commit**

```bash
uv run ruff check --fix napari_peredox/ tests/ && uv run ruff format napari_peredox/ tests/
git add napari_peredox/_host.py tests/test_host_assign.py
git commit -m "feat: add majority-overlap parasite-to-host assignment"
```

---

### Task 4: `_host.py` — `measure_hosts()`

**Files:**
- Modify: `napari_peredox/_host.py`
- Test: `tests/test_host_measure.py`

**Interfaces:**
- Consumes: `measure_pvs()` from `_measure.py`; `assign_to_hosts()` output dicts.
- Produces: `measure_hosts(host_labels, para_labels, image, para_to_host, vac_to_host, dilation_px=3, ch_cptsa=0, ch_mcherry=1, ch_names=None, pixel_size_um=None) -> pd.DataFrame` indexed by `host_id`, one row per host label in `host_labels`, with all `measure_pvs()` columns (measured on the cytosol = host minus dilated parasites) plus `infected` (bool), `n_parasites` (int), `n_vacuoles` (int), `parasite_area_px` (float), `host_area_px_total` (float), `on_border` (bool), `cytosol_empty` (bool). Fully-covered hosts keep their row with NaN measurement columns and `cytosol_empty=True`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_host_measure.py`:

```python
import numpy as np

from napari_peredox._host import measure_hosts


def _fixture():
    """
    30x30, 2-channel image.
    Host 1: rows 2-27, cols 2-13 (interior).   Host 2: rows 0-29, cols 16-29 (touches border).
    Parasite 1 (5x5) inside host 1.  Host 2 is uninfected.
    Channel 0 (cptsa) = 20 in hosts, 500 in parasite; channel 1 (mcherry) = 10 everywhere non-bg.
    """
    hosts = np.zeros((30, 30), dtype=np.int32)
    hosts[2:28, 2:14] = 1
    hosts[:, 16:30] = 2
    paras = np.zeros_like(hosts)
    paras[10:15, 5:10] = 1
    img = np.zeros((30, 30, 2), dtype=np.float32)
    img[..., 0][hosts > 0] = 20.0
    img[..., 1][hosts > 0] = 10.0
    img[..., 0][paras > 0] = 500.0
    return hosts, paras, img


def test_one_row_per_host_with_infection_flags():
    hosts, paras, img = _fixture()
    df = measure_hosts(
        hosts, paras, img, para_to_host={1: 1}, vac_to_host={7: 1}, dilation_px=2
    )
    assert sorted(df.index.tolist()) == [1, 2]
    assert df.index.name == "host_id"
    assert bool(df.loc[1, "infected"]) is True
    assert bool(df.loc[2, "infected"]) is False
    assert int(df.loc[1, "n_parasites"]) == 1
    assert int(df.loc[2, "n_parasites"]) == 0
    assert int(df.loc[1, "n_vacuoles"]) == 1
    assert df.loc[1, "parasite_area_px"] == 25.0


def test_parasite_pixels_plus_dilation_excluded_from_host_ratio():
    hosts, paras, img = _fixture()
    df = measure_hosts(
        hosts, paras, img, para_to_host={1: 1}, vac_to_host={}, dilation_px=2
    )
    # If any 500-valued parasite pixel leaked into host 1's cytosol, the mean
    # cptsa would exceed 20.  Ratio = 20/10 = 2 exactly when exclusion worked.
    assert abs(df.loc[1, "ratio_intden"] - 2.0) < 1e-6
    # Cytosol area shrank by MORE than the raw parasite area (dilation buffer)
    assert df.loc[1, "area_px"] < df.loc[1, "host_area_px_total"] - 25.0


def test_host_area_px_total_is_full_footprint():
    hosts, paras, img = _fixture()
    df = measure_hosts(hosts, paras, img, para_to_host={1: 1}, vac_to_host={})
    assert df.loc[1, "host_area_px_total"] == float(26 * 12)
    assert df.loc[2, "host_area_px_total"] == float(30 * 14)


def test_on_border_flag():
    hosts, paras, img = _fixture()
    df = measure_hosts(hosts, paras, img, para_to_host={}, vac_to_host={})
    assert bool(df.loc[1, "on_border"]) is False
    assert bool(df.loc[2, "on_border"]) is True


def test_fully_covered_host_kept_with_nan_ratio():
    hosts = np.zeros((10, 10), dtype=np.int32)
    hosts[2:6, 2:6] = 1
    paras = np.zeros_like(hosts)
    paras[2:6, 2:6] = 1  # parasite covers the entire host
    img = np.ones((10, 10, 2), dtype=np.float32)
    df = measure_hosts(hosts, paras, img, para_to_host={1: 1}, vac_to_host={})
    assert 1 in df.index
    assert bool(df.loc[1, "cytosol_empty"]) is True
    assert np.isnan(df.loc[1, "ratio_intden"])
    assert bool(df.loc[1, "infected"]) is True


def test_dilation_zero_excludes_exact_mask_only():
    hosts, paras, img = _fixture()
    df = measure_hosts(
        hosts, paras, img, para_to_host={1: 1}, vac_to_host={}, dilation_px=0
    )
    assert df.loc[1, "area_px"] == df.loc[1, "host_area_px_total"] - 25.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_host_measure.py -v`
Expected: FAIL — `ImportError: cannot import name 'measure_hosts'`

- [ ] **Step 3: Append the implementation to `_host.py`**

```python
def measure_hosts(
    host_labels: np.ndarray,
    para_labels: np.ndarray,
    image: np.ndarray,
    para_to_host: dict[int, int],
    vac_to_host: dict[int, int],
    dilation_px: int = 3,
    ch_cptsa: int = 0,
    ch_mcherry: int = 1,
    ch_names: dict[int, str] | None = None,
    pixel_size_um: float | None = None,
) -> pd.DataFrame:
    """
    Measure Peredox fluorescence per host cell on the parasite-free cytosol.

    The cytosol mask is the host mask minus all parasite pixels dilated by
    `dilation_px` (buffer against parasite signal bleed-over).  Ratios come
    from measure_pvs() run on that cytosol label image; host label IDs are
    preserved, so the returned index is the host ID.

    Every host in `host_labels` gets exactly one row.  A host whose cytosol
    vanishes entirely (heavy infection) keeps its row with NaN measurement
    columns and cytosol_empty=True.

    Added metadata columns (spec §3.1): infected, n_parasites, n_vacuoles,
    parasite_area_px, host_area_px_total, on_border, cytosol_empty.
    """
    from skimage.measure import regionprops
    from skimage.morphology import dilation, disk

    from ._measure import measure_pvs

    host_ids = sorted(int(v) for v in np.unique(host_labels) if v != 0)

    # ── Cytosol: host minus dilated parasite pixels ──────────────────────────
    cytosol = host_labels.copy()
    if para_labels.max() > 0:
        para_bin = para_labels > 0
        if dilation_px > 0:
            para_bin = dilation(para_bin, disk(dilation_px))
        cytosol[para_bin] = 0

    df = measure_pvs(cytosol, image, ch_cptsa, ch_mcherry, ch_names, pixel_size_um)

    # ── Full-footprint stats and border flags ────────────────────────────────
    H, W = host_labels.shape
    total_area: dict[int, float] = {}
    on_border: dict[int, bool] = {}
    for rp in regionprops(host_labels):
        total_area[int(rp.label)] = float(rp.area)
        minr, minc, maxr, maxc = rp.bbox
        on_border[int(rp.label)] = minr == 0 or minc == 0 or maxr == H or maxc == W

    # ── Parasite burden per host ─────────────────────────────────────────────
    n_para = dict.fromkeys(host_ids, 0)
    para_area = dict.fromkeys(host_ids, 0.0)
    for pid, hid in para_to_host.items():
        if hid in n_para:
            n_para[hid] += 1
            para_area[hid] += float((para_labels == pid).sum())
    n_vac = dict.fromkeys(host_ids, 0)
    for _vid, hid in vac_to_host.items():
        if hid in n_vac:
            n_vac[hid] += 1

    # ── Re-add hosts whose cytosol vanished entirely ─────────────────────────
    empty_ids = [h for h in host_ids if df.empty or h not in df.index]
    if empty_ids:
        nan_rows = pd.DataFrame(index=pd.Index(empty_ids, name="label"))
        df = pd.concat([df, nan_rows]) if not df.empty else nan_rows
    if "ratio_intden" not in df.columns:
        # Every host was fully covered — measure_pvs returned an empty frame,
        # so the ratio columns never materialised.  NaN keeps the schema.
        df["ratio_intden"] = np.nan
    df = df.loc[sorted(int(i) for i in df.index)]

    # ── Metadata columns ─────────────────────────────────────────────────────
    empty_set = set(empty_ids)
    df["cytosol_empty"] = [h in empty_set for h in df.index]
    df["infected"] = [n_para.get(h, 0) > 0 for h in df.index]
    df["n_parasites"] = [n_para.get(h, 0) for h in df.index]
    df["n_vacuoles"] = [n_vac.get(h, 0) for h in df.index]
    df["parasite_area_px"] = [para_area.get(h, 0.0) for h in df.index]
    df["host_area_px_total"] = [total_area.get(h, np.nan) for h in df.index]
    df["on_border"] = [on_border.get(h, False) for h in df.index]

    df.index.name = "host_id"
    return df
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_host_measure.py -v`
Expected: 6 passed.

- [ ] **Step 5: Ruff, then commit**

```bash
uv run ruff check --fix napari_peredox/ tests/ && uv run ruff format napari_peredox/ tests/
git add napari_peredox/_host.py tests/test_host_measure.py
git commit -m "feat: add per-host cytosol measurement with infection metadata"
```

---

### Task 5: `_host.py` — `segment_host_cells()`

**Files:**
- Modify: `napari_peredox/_host.py`
- Test: `tests/test_host_segment.py`

**Interfaces:**
- Consumes: `_segment._get_model()` (monkeypatched in tests), `_segment._filter_by_morphology()`, `clip_bright()`.
- Produces: `segment_host_cells(image, channel_index=1, clip_percentile=99.0, diameter=None, flow_threshold=0.4, cellprob_threshold=0.0, min_area_px=0.0, max_area_px=1e9) -> tuple[np.ndarray, np.ndarray, dict]` — same `(filtered, raw, stats)` convention as `segment_pvs()`. Shape gates are disabled (eccentricity 1.0, solidity 0.0): spread U2OS are irregular. `stats` additionally carries `clip_percentile`, `clip_value`, `clip_skipped`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_host_segment.py`:

```python
import numpy as np
import pytest

import napari_peredox._segment as _segment
from napari_peredox._host import segment_host_cells


class _FakeModel:
    """Records the image cpSAM would receive; returns a canned label image."""

    def __init__(self, labels):
        self._labels = labels
        self.last_input = None
        self.last_kwargs = None

    def eval(self, img, **kwargs):
        self.last_input = np.asarray(img).copy()
        self.last_kwargs = kwargs
        return self._labels, None, None


@pytest.fixture
def fake_model(monkeypatch):
    labels = np.zeros((20, 20), dtype=np.int32)
    labels[2:10, 2:10] = 1    # 64 px
    labels[12:14, 12:14] = 2  # 4 px
    model = _FakeModel(labels)
    monkeypatch.setattr(_segment, "_get_model", lambda: model)
    return model


def _img():
    # Graded background (5→20) so clipping never flattens the image entirely
    grad = np.linspace(5.0, 20.0, 400, dtype=np.float32).reshape(20, 20)
    img = np.stack([grad, grad.copy()], axis=-1)
    img[0, 0, 1] = 10000.0  # one very bright "parasite" pixel on channel 1
    return img


def test_clipping_applied_before_model(fake_model):
    segment_host_cells(_img(), channel_index=1, clip_percentile=99.0)
    assert fake_model.last_input.max() < 10000.0


def test_area_gate_filters_small_objects(fake_model):
    filtered, raw, stats = segment_host_cells(
        _img(), channel_index=1, min_area_px=10.0
    )
    assert set(np.unique(raw)) == {0, 1, 2}
    assert 2 not in np.unique(filtered)  # 4 px object rejected
    assert stats["kept"] == 1
    assert stats["rejected_area"] == 1


def test_stats_carry_clip_info(fake_model):
    _, _, stats = segment_host_cells(_img(), channel_index=1, clip_percentile=99.0)
    assert stats["clip_percentile"] == 99.0
    assert stats["clip_skipped"] is False
    assert stats["clip_value"] > 0


def test_flat_clip_falls_back_to_unclipped(fake_model):
    # A 2-value image where clipping at a low percentile flattens it entirely
    img = np.zeros((20, 20, 2), dtype=np.float32)
    img[..., 1] = 5.0  # constant non-zero channel: clip leaves it flat but equal
    _, _, stats = segment_host_cells(img, channel_index=1, clip_percentile=50.0)
    # Constant image: clipped max == min -> fallback path
    assert stats["clip_skipped"] is True
    np.testing.assert_array_equal(
        fake_model.last_input, img[..., 1]
    )


def test_2d_image_accepted(fake_model):
    img2d = np.full((20, 20), 10.0, dtype=np.float32)
    filtered, _, _ = segment_host_cells(img2d, channel_index=0)
    assert filtered.shape == (20, 20)


def test_model_receives_cpsam_kwargs(fake_model):
    segment_host_cells(
        _img(), channel_index=1, diameter=120.0,
        flow_threshold=0.5, cellprob_threshold=-1.0,
    )
    assert fake_model.last_kwargs["diameter"] == 120.0
    assert fake_model.last_kwargs["flow_threshold"] == 0.5
    assert fake_model.last_kwargs["cellprob_threshold"] == -1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_host_segment.py -v`
Expected: FAIL — `ImportError: cannot import name 'segment_host_cells'`

- [ ] **Step 3: Append the implementation to `_host.py`**

```python
def segment_host_cells(
    image: np.ndarray,
    channel_index: int = 1,
    clip_percentile: float = 99.0,
    diameter: float | None = None,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
    min_area_px: float = 0.0,
    max_area_px: float = 1e9,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Stage H1: segment host cells with cpSAM on a bright-clipped channel.

    Whole-cell segmentation is cpSAM's native task; the only special handling
    is clip_bright(), which stops much-brighter intracellular parasites from
    compressing host contrast during model normalisation.  Only the area gate
    is applied afterwards — spread U2OS fail the PV eccentricity/solidity
    gates, so those are disabled (spec §4 H1).

    Note: cpSAM may propose a bright vacuole inside a NON-expressing cell as a
    small "host" — the min-area gate is what rejects these (spec §4 H1), so
    keep min_area_px well above any vacuole footprint.

    Returns
    -------
    (filtered_labels, raw_labels, stats) — same convention as segment_pvs().
    stats additionally has clip_percentile, clip_value, clip_skipped.
    """
    from ._segment import _filter_by_morphology, _get_model

    if image.ndim == 3:
        chan = image[..., channel_index].astype(np.float32)
    elif image.ndim == 2:
        chan = image.astype(np.float32)
    else:
        raise ValueError(f"Expected 2-D or 3-D array, got shape {image.shape}")

    clipped = clip_bright(chan, clip_percentile)
    if clipped.max() > clipped.min():
        seg_img = clipped
        clip_skipped = False
    else:
        # Clipping flattened the image (or it was already flat) — cpSAM would
        # see no contrast, so fall back to the unclipped channel.
        seg_img = chan
        clip_skipped = True

    model = _get_model()
    masks, _, _ = model.eval(
        seg_img,
        diameter=diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        normalize=True,
    )
    raw_labels = np.asarray(masks).astype(np.int32)

    filtered, stats = _filter_by_morphology(
        raw_labels, min_area_px, max_area_px, 1.0, 0.0
    )
    stats.update(
        {
            "clip_percentile": float(clip_percentile),
            "clip_value": float(seg_img.max()),
            "clip_skipped": clip_skipped,
        }
    )
    return filtered, raw_labels, stats
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_host_segment.py -v`
Expected: 6 passed.

- [ ] **Step 5: Run the whole suite, ruff, commit**

```bash
uv run pytest tests/ -v
uv run ruff check --fix napari_peredox/ tests/ && uv run ruff format napari_peredox/ tests/
git add napari_peredox/_host.py tests/test_host_segment.py
git commit -m "feat: add cpSAM host-cell segmentation with bright-pixel clipping"
```

---

### Task 6: host-aware paths in `_io.py` and `_learning.py`

**Files:**
- Modify: `napari_peredox/_io.py:101-141` (`append_curated_annotations`)
- Modify: `napari_peredox/_learning.py:254-266` (`load_classifier`)
- Test: `tests/test_host_io.py`

**Interfaces:**
- Produces: `append_curated_annotations(..., csv_name="curated_features.csv")` — same behavior, new keyword selects the target CSV (host mode passes `"curated_host_features.csv"`).
- Produces: `load_classifier(annotations_dir, filename="curated_features.joblib")` — same behavior, new keyword selects the model file (host mode passes `"curated_host_features.joblib"`).
- Both defaults preserve every existing call site unchanged.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_host_io.py`:

```python
import pandas as pd

from napari_peredox._io import append_curated_annotations


def test_default_csv_name_unchanged(tmp_path):
    path = append_curated_annotations(
        decisions={1: 1}, features=pd.DataFrame(), image_stem="img",
        annotations_dir=tmp_path,
    )
    assert path.name == "curated_features.csv"
    assert path.exists()


def test_custom_csv_name_writes_host_file(tmp_path):
    path = append_curated_annotations(
        decisions={1: 1, 2: 0}, features=pd.DataFrame(), image_stem="img",
        annotations_dir=tmp_path, csv_name="curated_host_features.csv",
    )
    assert path.name == "curated_host_features.csv"
    df = pd.read_csv(path)
    assert len(df) == 2
    assert not (tmp_path / "curated_features.csv").exists()


def test_load_classifier_custom_filename(tmp_path):
    from napari_peredox._learning import load_classifier

    # No model files exist -> both return None, but neither must raise
    assert load_classifier(tmp_path) is None
    assert load_classifier(tmp_path, filename="curated_host_features.joblib") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_host_io.py -v`
Expected: `test_default_csv_name_unchanged` PASSES (existing behavior); the other two FAIL with `TypeError: ... unexpected keyword argument`.

- [ ] **Step 3: Make the two edits**

In `_io.py`, change the signature of `append_curated_annotations` and its one hard-coded path line:

```python
def append_curated_annotations(
    decisions: dict[int, int],
    features: pd.DataFrame,
    image_stem: str,
    annotations_dir: str | Path,
    vacuole_assignments: dict[int, int] | None = None,
    csv_name: str = "curated_features.csv",
) -> Path:
```

and inside, replace `csv_path = annotations_dir / "curated_features.csv"` with:

```python
    csv_path = annotations_dir / csv_name
```

In `_learning.py`, change `load_classifier`:

```python
def load_classifier(
    annotations_dir: str | Path,
    filename: str = "curated_features.joblib",
) -> object | None:
    """
    Load a previously trained classifier from the annotations directory.

    `filename` selects which model to load — the PV classifier by default,
    or "curated_host_features.joblib" for the host-cell classifier.
    Returns None if no model file is found.
    """
    import joblib

    model_path = Path(annotations_dir) / filename
    if model_path.exists():
        return joblib.load(model_path)
    return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/ -v`
Expected: all pass (including all earlier suites — this guards the default-path behavior).

- [ ] **Step 5: Ruff, then commit**

```bash
uv run ruff check --fix napari_peredox/ tests/ && uv run ruff format napari_peredox/ tests/
git add napari_peredox/_io.py napari_peredox/_learning.py tests/test_host_io.py
git commit -m "feat: parameterize annotation CSV and classifier filenames for host mode"
```

---

### Task 7: `VacuoleCurationWidget` object-name parameterization

**Files:**
- Modify: `napari_peredox/_curation.py:396-682` (`VacuoleCurationWidget`)
- Test: `tests/test_curation_object_name.py`

**Interfaces:**
- Produces: `VacuoleCurationWidget(..., object_name: str = "vacuole")` — the word shown in nav label, draw-button hints, and the save button. No behavioral change; host mode passes `object_name="host cell"`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_curation_object_name.py`:

```python
"""Signature-level test: _curation imports Qt, so avoid instantiating widgets."""

import inspect


def test_vacuole_curation_widget_accepts_object_name():
    from napari_peredox._curation import VacuoleCurationWidget

    sig = inspect.signature(VacuoleCurationWidget.__init__)
    assert "object_name" in sig.parameters
    assert sig.parameters["object_name"].default == "vacuole"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_curation_object_name.py -v`
Expected: FAIL — `assert "object_name" in sig.parameters`.
(If the import itself fails because Qt needs a display, run with `QT_QPA_PLATFORM=offscreen`: `$env:QT_QPA_PLATFORM='offscreen'; uv run pytest ...` — record whichever works in NOTES for later tasks.)

- [ ] **Step 3: Implement**

In `VacuoleCurationWidget.__init__`, add the parameter after `labels_layer_name`:

```python
        labels_layer_name: str | None = None,
        object_name: str = "vacuole",
        parent=None,
    ):
        super().__init__(parent)
        self._object_name = object_name
```

Then replace the user-facing strings (all inside this class only):

- `_build_ui`: `self._btn_save = QPushButton("💾 Save & retrain vacuole model")` → `QPushButton(f"💾 Save {self._object_name} review")` — keep the string exactly `"💾 Save & retrain vacuole model"` when `object_name == "vacuole"` so the PV workflow is untouched:

```python
        if self._object_name == "vacuole":
            self._btn_save = QPushButton("💾 Save & retrain vacuole model")
        else:
            self._btn_save = QPushButton(f"💾 Save {self._object_name} review")
```

- `_build_ui` draw hint and `_cancel_draw`/`_on_polygon_closed` reset strings: replace the literal `"Click ✏ Draw outline to redraw the vacuole boundary."` (3 occurrences) with `f"Click ✏ Draw outline to redraw the {self._object_name} boundary."`
- `_refresh`: `self._lbl_nav.setText(f"Vacuole {idx + 1} / {n}  (id={vac_id})")` → `f"{self._object_name.capitalize()} {idx + 1} / {n}  (id={vac_id})"`, and the empty-state `"No vacuoles"` → `f"No {self._object_name}s"`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_curation_object_name.py -v`
Expected: PASS.

- [ ] **Step 5: Ruff, import check, commit**

```bash
uv run ruff check --fix napari_peredox/ tests/ && uv run ruff format napari_peredox/ tests/
uv run python -c "from napari_peredox._curation import VacuoleCurationWidget; print('OK')"
git add napari_peredox/_curation.py tests/test_curation_object_name.py
git commit -m "feat: parameterize curation gallery object name for host review"
```

---

### Task 8: Host tab UI + `_HostWorker` (Stage H1)

**Files:**
- Modify: `napari_peredox/_widget.py` — new worker class after `_ParasiteWorker` (after line ~282), new state attrs in `PeredoxWidget.__init__`, new tab in `_build_ui`, new methods at the end of the class.

**Interfaces:**
- Consumes: `segment_host_cells()`, `clip_bright()` (Task 5), `extract_features()`, `apply_classifier_filter()`, `load_classifier(annot_dir, filename="curated_host_features.joblib")` (Task 6).
- Produces: `PeredoxWidget._host_labels: np.ndarray | None`; UI attrs `_host_ch`, `_host_clip_pct`, `_host_diameter`, `_host_min_area_um2`, `_host_max_area_um2`, `_host_dilation_px`, `_host_use_classifier`, buttons `_btn_host_stage1/_btn_host_review/_btn_host_stage2/_btn_host_review_para/_btn_host_measure/_btn_host_export`, labels `_lbl_host_stage1/_lbl_host_stage2/_lbl_host_measure`; methods `_run_host_stage1()`, `_on_host_stage1_done()`, `_host_pixel_area_limits()`. Later tasks (9-11) fill in the review/stage2/measure handlers — this task wires their buttons to placeholder methods that just log, so the tab is fully clickable without crashing.

- [ ] **Step 1: Add the worker class** (after `_ParasiteWorker`, before "Main widget" comment block):

```python
# ---------------------------------------------------------------------------
# Background worker — Stage H1: host-cell segmentation
# ---------------------------------------------------------------------------


class _HostWorker(QObject):
    """Run host-cell segmentation (cpSAM + clip) in a background thread."""

    finished = Signal(object, object)  # (raw_host_labels, host_labels)
    error = Signal(str)
    progress = Signal(str)

    def __init__(self, params: dict):
        super().__init__()
        self.params = params

    def run(self) -> None:
        try:
            p = self.params
            from ._host import segment_host_cells

            self.progress.emit("Running cpSAM host-cell segmentation…")
            host_labels, raw_labels, stats = segment_host_cells(
                image=p["image"],
                channel_index=p["host_ch"],
                clip_percentile=p["clip_percentile"],
                diameter=p.get("diameter"),
                flow_threshold=p.get("flow_threshold", 0.4),
                cellprob_threshold=p.get("cellprob_threshold", 0.0),
                min_area_px=p["min_area_px"],
                max_area_px=p["max_area_px"],
            )
            if stats.get("clip_skipped"):
                self.progress.emit(
                    "WARNING: bright-pixel clip flattened the image — "
                    "segmented the unclipped channel instead."
                )
            self.progress.emit(
                f"cpSAM hosts: {stats['total_raw']} raw → {stats['kept']} "
                f"after area gate (clip @ p{stats['clip_percentile']:.1f})."
            )

            # Optional host classifier filter (spec §6)
            classifier = p.get("classifier")
            if classifier is not None and host_labels.max() > 0:
                from ._learning import extract_features
                from ._segment import apply_classifier_filter

                feats = extract_features(
                    labels=host_labels,
                    image=p["image"],
                    seg_channel=p["host_ch"],
                    ch_cptsa=p["ch_cptsa"],
                    ch_mcherry=p["ch_mcherry"],
                    ch_names=p["ch_names"],
                )
                n_before = int(len(feats))
                host_labels = apply_classifier_filter(host_labels, feats, classifier)
                self.progress.emit(
                    f"Host classifier: {n_before} → "
                    f"{int(host_labels.max())} hosts kept."
                )

            self.finished.emit(raw_labels, host_labels)
        except Exception:
            import traceback

            self.error.emit(traceback.format_exc())
```

- [ ] **Step 2: Add host state to `PeredoxWidget.__init__`** (after the Stage-2 state block):

```python
        # Host-mode state (Host tab) — never shared with PV-mode state above
        self._host_labels: np.ndarray | None = None
        self._host_para_labels: np.ndarray | None = None
        self._host_vac_map: dict = {}
        self._host_features = None
        self._host_para_measurements = None
        self._host_measurements = None
        self._host_classifier = None
```

- [ ] **Step 3: Register the tab in `_build_ui`:**

```python
        tabs.addTab(self._build_segment_tab(), "Segment")
        tabs.addTab(self._build_host_tab(), "Host")
        tabs.addTab(self._build_training_tab(), "Training")
```

- [ ] **Step 4: Add the tab builder and Stage-H1 methods** at the end of the class (before `make_main_widget`):

```python
    # ── Host tab ─────────────────────────────────────────────────────────────

    def _build_host_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # Host segmentation parameters
        par_box = QGroupBox("Host segmentation (cpSAM)")
        par_form = QFormLayout(par_box)
        par_form.setContentsMargins(6, 6, 6, 6)

        self._host_ch = QSpinBox()
        self._host_ch.setRange(0, 15)
        self._host_ch.setValue(1)
        self._host_ch.setToolTip(
            "Channel for host-cell segmentation (default: mCherry)."
        )
        par_form.addRow("Host channel:", self._host_ch)

        self._host_clip_pct = QDoubleSpinBox()
        self._host_clip_pct.setRange(50.0, 100.0)
        self._host_clip_pct.setDecimals(1)
        self._host_clip_pct.setSingleStep(0.5)
        self._host_clip_pct.setValue(99.0)
        self._host_clip_pct.setToolTip(
            "Clip pixels above this percentile before cpSAM so bright\n"
            "parasites don't flatten host contrast. 100 = no clipping."
        )
        par_form.addRow("Clip percentile:", self._host_clip_pct)

        self._host_diameter = QSpinBox()
        self._host_diameter.setRange(0, 2000)
        self._host_diameter.setValue(0)
        self._host_diameter.setSpecialValueText("auto")
        self._host_diameter.setToolTip("Expected host-cell diameter in px (0 = auto).")
        par_form.addRow("Diameter (px):", self._host_diameter)

        self._host_min_area_um2 = QDoubleSpinBox()
        self._host_min_area_um2.setRange(0.0, 1e6)
        self._host_min_area_um2.setDecimals(0)
        self._host_min_area_um2.setValue(200.0)
        self._host_max_area_um2 = QDoubleSpinBox()
        self._host_max_area_um2.setRange(0.0, 1e6)
        self._host_max_area_um2.setDecimals(0)
        self._host_max_area_um2.setValue(10000.0)
        area_row = QHBoxLayout()
        area_row.addWidget(QLabel("min:"))
        area_row.addWidget(self._host_min_area_um2)
        area_row.addWidget(QLabel("max:"))
        area_row.addWidget(self._host_max_area_um2)
        par_form.addRow("Host area (µm²):", area_row)

        self._host_dilation_px = QSpinBox()
        self._host_dilation_px.setRange(0, 50)
        self._host_dilation_px.setValue(3)
        self._host_dilation_px.setToolTip(
            "Dilate parasite masks by this many px before excluding them\n"
            "from the host ratio (buffer against signal bleed-over)."
        )
        par_form.addRow("Parasite exclusion buffer (px):", self._host_dilation_px)

        self._host_use_classifier = QCheckBox("Apply host classifier filter")
        par_form.addRow("", self._host_use_classifier)

        layout.addWidget(par_box)

        # Stage H1
        h1_box = QGroupBox("Stage H1 — Host cells")
        h1_layout = QVBoxLayout(h1_box)
        self._btn_host_stage1 = QPushButton("▶ Segment host cells")
        self._btn_host_stage1.setStyleSheet("font-weight: bold;")
        self._btn_host_stage1.clicked.connect(self._run_host_stage1)
        h1_layout.addWidget(self._btn_host_stage1)
        self._lbl_host_stage1 = QLabel("Not run yet.")
        self._lbl_host_stage1.setWordWrap(True)
        h1_layout.addWidget(self._lbl_host_stage1)
        self._btn_host_review = QPushButton("🔍 Review host cells")
        self._btn_host_review.clicked.connect(self._open_host_curation)
        self._btn_host_review.setEnabled(False)
        h1_layout.addWidget(self._btn_host_review)
        layout.addWidget(h1_box)

        # Stage H2
        h2_box = QGroupBox("Stage H2 — Parasites in hosts")
        h2_layout = QVBoxLayout(h2_box)
        self._btn_host_stage2 = QPushButton("▶ Segment parasites in hosts")
        self._btn_host_stage2.setStyleSheet("font-weight: bold;")
        self._btn_host_stage2.clicked.connect(self._run_host_stage2)
        self._btn_host_stage2.setEnabled(False)
        h2_layout.addWidget(self._btn_host_stage2)
        self._lbl_host_stage2 = QLabel("Segment and review host cells first.")
        self._lbl_host_stage2.setWordWrap(True)
        h2_layout.addWidget(self._lbl_host_stage2)
        self._btn_host_review_para = QPushButton("🔍 Review parasites")
        self._btn_host_review_para.clicked.connect(self._open_host_parasite_curation)
        self._btn_host_review_para.setEnabled(False)
        h2_layout.addWidget(self._btn_host_review_para)
        layout.addWidget(h2_box)

        # Stage H3
        h3_box = QGroupBox("Stage H3 — Measure && export")
        h3_layout = QVBoxLayout(h3_box)
        self._btn_host_measure = QPushButton("▶ Assign, measure && show table")
        self._btn_host_measure.setStyleSheet("font-weight: bold;")
        self._btn_host_measure.clicked.connect(self._run_host_measure)
        self._btn_host_measure.setEnabled(False)
        h3_layout.addWidget(self._btn_host_measure)
        self._lbl_host_measure = QLabel("Complete Stages H1–H2 first.")
        self._lbl_host_measure.setWordWrap(True)
        h3_layout.addWidget(self._lbl_host_measure)
        self._btn_host_export = QPushButton("💾 Export host && parasite CSVs")
        self._btn_host_export.clicked.connect(self._export_host_csv)
        self._btn_host_export.setEnabled(False)
        h3_layout.addWidget(self._btn_host_export)
        layout.addWidget(h3_box)

        layout.addStretch()
        return w

    def _host_pixel_area_limits(self) -> tuple[float, float]:
        px = self._pixel_size.value()
        if px > 0:
            return (
                self._host_min_area_um2.value() / (px**2),
                self._host_max_area_um2.value() / (px**2),
            )
        return 0.0, 1e9

    # ── Stage H1: host-cell segmentation ─────────────────────────────────────

    def _run_host_stage1(self) -> None:
        try:
            image = self._get_image_array()
        except RuntimeError as exc:
            self._log_msg(f"Error: {exc}")
            return

        n_ch = image.shape[-1]
        ch_names = {i: f"ch{i}" for i in range(n_ch)}
        ch_names[self._ch_cptsa.value()] = "cptsa"
        ch_names[self._ch_mcherry.value()] = "mcherry"
        self._image_stem = self._layer_combo.currentText().replace(" ", "_") or "image"

        if self._pixel_size.value() == 0:
            val, ok = QInputDialog.getDouble(
                self,
                "Pixel size not set",
                "Enter physical pixel size (µm/px).\n"
                "Cancel or enter 0 to disable the host area filter:",
                decimals=4,
                min=0.0,
                max=100.0,
                value=0.105,
            )
            if ok and val > 0:
                self._pixel_size.setValue(val)

        min_area_px, max_area_px = self._host_pixel_area_limits()

        host_classifier = None
        if self._host_use_classifier.isChecked():
            from ._learning import load_classifier

            host_classifier = load_classifier(
                self._annot_dir.text(), filename="curated_host_features.joblib"
            )
            if host_classifier is None:
                self._log_msg("No host classifier trained yet — running without.")

        diameter = self._host_diameter.value()
        params = {
            "image": image,
            "host_ch": self._host_ch.value(),
            "clip_percentile": self._host_clip_pct.value(),
            "diameter": float(diameter) if diameter > 0 else None,
            "flow_threshold": self._flow_thresh.value(),
            "cellprob_threshold": self._cellprob_thresh.value(),
            "min_area_px": min_area_px,
            "max_area_px": max_area_px,
            "classifier": host_classifier,
            "ch_cptsa": self._ch_cptsa.value(),
            "ch_mcherry": self._ch_mcherry.value(),
            "ch_names": ch_names,
        }

        self._btn_host_stage1.setEnabled(False)
        self._btn_host_stage1.setText("Running…")
        self._lbl_host_stage1.setText("Segmenting host cells…")

        from ._segment import preload_model

        self._log_msg(preload_model())

        self._thread = QThread()
        self._worker = _HostWorker(params)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_host_stage1_done)
        self._worker.error.connect(self._on_host_worker_error)
        self._worker.progress.connect(self._log_msg)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.start()

    def _on_host_stage1_done(
        self, raw_host_labels: np.ndarray, host_labels: np.ndarray
    ) -> None:
        self._host_labels = host_labels
        # Reset downstream host state — new hosts invalidate old parasites
        self._host_para_labels = None
        self._host_vac_map = {}
        self._host_measurements = None
        stem = self._image_stem

        raw_name = f"{stem}_host_candidates"
        if raw_name in self._viewer.layers:
            self._viewer.layers[raw_name].data = raw_host_labels
        else:
            self._viewer.add_labels(raw_host_labels, name=raw_name, opacity=0.2)

        host_name = f"{stem}_hosts"
        if host_name in self._viewer.layers:
            self._viewer.layers[host_name].data = host_labels
        else:
            self._viewer.add_labels(host_labels, name=host_name)

        n = len(np.unique(host_labels)) - 1
        self._lbl_host_stage1.setText(f"Found {n} host cells.")
        self._log_msg(f"Stage H1 done — {n} host cells.")
        self._btn_host_stage1.setEnabled(True)
        self._btn_host_stage1.setText("▶ Segment host cells")
        self._btn_host_review.setEnabled(True)
        self._btn_host_stage2.setEnabled(n > 0)
        if n == 0:
            self._lbl_host_stage2.setText("No hosts found — Stage H2 unavailable.")

    def _on_host_worker_error(self, msg: str) -> None:
        self._log_msg(f"Error:\n{msg}")
        self._btn_host_stage1.setEnabled(True)
        self._btn_host_stage1.setText("▶ Segment host cells")
        self._btn_host_stage2.setEnabled(self._host_labels is not None)
        self._btn_host_stage2.setText("▶ Segment parasites in hosts")

    # ── Placeholders completed in Tasks 9–11 ─────────────────────────────────

    def _open_host_curation(self) -> None:
        self._log_msg("Host curation not implemented yet (Task 9).")

    def _run_host_stage2(self) -> None:
        self._log_msg("Stage H2 not implemented yet (Task 10).")

    def _open_host_parasite_curation(self) -> None:
        self._log_msg("Host parasite curation not implemented yet (Task 10).")

    def _run_host_measure(self) -> None:
        self._log_msg("Stage H3 not implemented yet (Task 11).")

    def _export_host_csv(self) -> None:
        self._log_msg("Host export not implemented yet (Task 11).")
```

- [ ] **Step 5: Ruff, import check, commit**

```bash
uv run ruff check --fix napari_peredox/ tests/ && uv run ruff format napari_peredox/ tests/
uv run python -c "from napari_peredox import _widget; print('OK')"
uv run pytest tests/ -v
git add napari_peredox/_widget.py
git commit -m "feat: add Host tab with Stage H1 host segmentation worker"
```

---

### Task 9: host curation wiring (review → host classifier)

**Files:**
- Modify: `napari_peredox/_widget.py` — replace the `_open_host_curation` placeholder; add `_on_host_curation_saved`.

**Interfaces:**
- Consumes: `VacuoleCurationWidget(object_name="host cell")` (Task 7), `append_curated_annotations(csv_name="curated_host_features.csv")` (Task 6), `extract_features()`, `train_classifier()`.
- Produces: after save, `self._host_labels` holds accepted hosts only (rejected zeroed; polygon edits applied) and the host classifier retrains from `curated_host_features.csv` (model saved by `train_classifier` as `curated_host_features.joblib`).

- [ ] **Step 1: Replace the placeholder methods**

```python
    # ── Host curation ────────────────────────────────────────────────────────

    def _open_host_curation(self) -> None:
        if self._host_labels is None:
            self._log_msg("Run Stage H1 first.")
            return
        try:
            image = self._get_image_array()
        except RuntimeError:
            image = np.zeros((*self._host_labels.shape, 2), dtype=np.float32)

        from ._curation import VacuoleCurationWidget

        host_layer_name = f"{self._image_stem}_hosts"
        self._host_curation_win = VacuoleCurationWidget(
            vac_labels=self._host_labels,
            image=image,
            ch_cptsa=self._ch_cptsa.value(),
            ch_mcherry=self._ch_mcherry.value(),
            on_save=self._on_host_curation_saved,
            viewer=self._viewer,
            labels_layer_name=host_layer_name,
            object_name="host cell",
            parent=None,
        )
        self._host_curation_win.setWindowTitle("Peredox — Host Cell Review (Stage H1)")
        self._host_curation_win.resize(360, 560)
        self._host_curation_win.show()

    def _on_host_curation_saved(
        self, decisions: dict, curated_host_labels: np.ndarray
    ) -> None:
        """Persist host accept/reject decisions and retrain the host classifier."""
        self._host_labels = curated_host_labels.copy()

        # Grow the host training CSV and retrain (spec §6) — separate files
        # from the PV classifier, same machinery.
        try:
            image = self._get_image_array()
            from ._io import append_curated_annotations
            from ._learning import extract_features, train_classifier

            n_ch = image.shape[-1]
            ch_names = {i: f"ch{i}" for i in range(n_ch)}
            ch_names[self._ch_cptsa.value()] = "cptsa"
            ch_names[self._ch_mcherry.value()] = "mcherry"

            feats = extract_features(
                labels=curated_host_labels,
                image=image,
                seg_channel=self._host_ch.value(),
                ch_cptsa=self._ch_cptsa.value(),
                ch_mcherry=self._ch_mcherry.value(),
                ch_names=ch_names,
            )
            csv_path = append_curated_annotations(
                decisions=decisions,
                features=feats,
                image_stem=self._image_stem,
                annotations_dir=self._annot_dir.text(),
                csv_name="curated_host_features.csv",
            )
            n_dec = sum(1 for v in decisions.values() if v in (0, 1))
            self._log_msg(f"Saved {n_dec} host annotations → {csv_path}")

            clf = train_classifier(csv_path)
            if clf is not None:
                self._host_classifier = clf
                self._log_msg("Host classifier retrained.")
            else:
                self._log_msg("Not enough host data to train the classifier yet.")
        except Exception as exc:
            self._log_msg(f"Host annotation save error: {exc}")

        # Drop rejected hosts from the working label image
        rejected = {hid for hid, dec in decisions.items() if dec == 0}
        for hid in rejected:
            self._host_labels[self._host_labels == hid] = 0

        n_remaining = len(np.unique(self._host_labels)) - 1
        self._log_msg(f"Host curation saved — {n_remaining} hosts kept.")
        self._lbl_host_stage1.setText(
            f"Curation done — {n_remaining} accepted hosts ready for Stage H2."
        )
        self._btn_host_stage2.setEnabled(n_remaining > 0)
        if n_remaining > 0:
            self._lbl_host_stage2.setText("Ready — click to detect parasites.")
        # Invalidate any parasites detected against the pre-curation hosts
        self._host_para_labels = None
        self._btn_host_review_para.setEnabled(False)
        self._btn_host_measure.setEnabled(False)
```

Note: `VacuoleCurationWidget._save` treats unreviewed hosts as accepted, so `decisions` arrives with `-1` already promoted to `1`; `n_accepted` is informational only.

- [ ] **Step 2: Ruff, import check, run suite, commit**

```bash
uv run ruff check --fix napari_peredox/ tests/ && uv run ruff format napari_peredox/ tests/
uv run python -c "from napari_peredox import _widget; print('OK')"
uv run pytest tests/ -v
git add napari_peredox/_widget.py
git commit -m "feat: wire host curation gallery with host-classifier training"
```

---

### Task 10: Stage H2 — parasites inside accepted hosts

**Files:**
- Modify: `napari_peredox/_widget.py` — new `_HostParasiteWorker` class after `_HostWorker`; replace `_run_host_stage2` and `_open_host_parasite_curation` placeholders; add `_on_host_stage2_done` and `_on_host_parasite_curation_saved`.

**Interfaces:**
- Consumes: `segment_pvs()`, `segment_parasites_in_vacuoles()`, `load_stardist_model()`, `extract_features()`, `measure_pvs()`, `CurationWidget`, `append_curated_annotations()` (default CSV — parasite appearance is the same task as PV mode, spec §4), `train_classifier()`.
- Produces: `self._host_para_labels` (int32 parasite labels, only inside accepted hosts), `self._host_vac_map` (parasite label → vacuole ID), `self._host_features`, `self._host_para_measurements` (per-parasite `measure_pvs()` table — **not** vacuole-reduced; Stage H3 joins host IDs onto it).
- Stage H2 reads the **Setup tab's** segmentation settings (backend, thresholds, PV-scale area/shape gates) — the Host tab adds no duplicate controls for parasite detection (spec §4: existing pipeline, unchanged).

- [ ] **Step 1: Add the worker class** (after `_HostWorker`):

```python
# ---------------------------------------------------------------------------
# Background worker — Stage H2: parasites within accepted hosts
# ---------------------------------------------------------------------------


class _HostParasiteWorker(QObject):
    """Vacuole→parasite detection on the host-masked image (spec §4 H2)."""

    finished = Signal(object, object, object, object)
    # (para_labels, vacuole_map, features, para_measurements)
    error = Signal(str)
    progress = Signal(str)

    def __init__(self, params: dict):
        super().__init__()
        self.params = params

    def run(self) -> None:
        try:
            p = self.params
            image = p["image"]
            host_labels = p["host_labels"]
            seg_ch = p["seg_ch"]
            seg_backend = p.get("seg_backend", 0)
            annot_dir = p.get("annot_dir", "")

            # Zero the image outside accepted hosts: extracellular parasites
            # and parasites in non-expressing cells are excluded by
            # construction (spec §2 population rule).
            host_union = host_labels > 0
            masked = image * host_union[..., np.newaxis].astype(image.dtype)

            # ── Vacuole detection on the masked image ────────────────────────
            if seg_backend == 1:
                from pathlib import Path as _Path

                from ._segment import filter_labels
                from ._stardist import load_stardist_model, predict_stardist

                model_dir = _Path(annot_dir) / "stardist_model"
                self.progress.emit("Loading vacuole StarDist model…")
                sd_vac = load_stardist_model(model_dir, mode="vacuoles")
                if sd_vac is None:
                    raise RuntimeError(
                        "Vacuole StarDist model not found — train it first "
                        "or switch the Setup tab to cpSAM."
                    )
                raw_vac = predict_stardist(masked, sd_vac, seg_ch)
                vac_labels, _, fstats = filter_labels(
                    raw_vac,
                    p["vac_min_area_px"],
                    p["vac_max_area_px"],
                    p["max_eccentricity"],
                    p["min_solidity"],
                )
            else:
                from ._segment import segment_pvs

                self.progress.emit("Running cpSAM vacuole detection in hosts…")
                vac_labels, _, fstats = segment_pvs(
                    image=masked,
                    channel_index=seg_ch,
                    use_composite=p.get("use_composite", False),
                    min_area_px=p["vac_min_area_px"],
                    max_area_px=p["vac_max_area_px"],
                    max_eccentricity=p["max_eccentricity"],
                    min_solidity=p["min_solidity"],
                    diameter=p.get("diameter"),
                    flow_threshold=p.get("flow_threshold", 0.4),
                    cellprob_threshold=p.get("cellprob_threshold", 0.0),
                    threshold_method=p.get("threshold_method", "none"),
                    threshold_channel=p.get("threshold_channel", seg_ch),
                    threshold_value=p.get("threshold_value", 0.0),
                    threshold_percentile=p.get("threshold_percentile", 50.0),
                )
            self.progress.emit(
                f"Vacuoles in hosts: {fstats['total_raw']} raw → "
                f"{fstats['kept']} after filter."
            )

            # ── Parasites within those vacuoles ──────────────────────────────
            sd_para = None
            if seg_backend == 1:
                from pathlib import Path as _Path

                from ._stardist import load_stardist_model

                sd_para = load_stardist_model(
                    _Path(annot_dir) / "stardist_model", mode="parasites"
                )
                if sd_para is None:
                    self.progress.emit(
                        "Parasite StarDist model not found — using cpSAM fallback."
                    )

            from ._segment import segment_parasites_in_vacuoles

            para_labels, vacuole_map = segment_parasites_in_vacuoles(
                image=masked,
                vac_labels=vac_labels,
                seg_channel=seg_ch,
                model=sd_para,
                min_area_px=p["min_area_px"],
                max_area_px=p["max_area_px"],
                max_eccentricity=p["max_eccentricity"],
                min_solidity=p["min_solidity"],
                progress_cb=self.progress.emit,
            )

            from ._learning import extract_features

            features = extract_features(
                labels=para_labels,
                image=image,
                seg_channel=seg_ch,
                ch_cptsa=p["ch_cptsa"],
                ch_mcherry=p["ch_mcherry"],
                ch_names=p["ch_names"],
            )

            from ._measure import measure_pvs

            self.progress.emit("Measuring parasites…")
            pixel_size = p.get("pixel_size", 0.0)
            para_measurements = measure_pvs(
                labels=para_labels,
                image=image,
                ch_cptsa=p["ch_cptsa"],
                ch_mcherry=p["ch_mcherry"],
                ch_names=p["ch_names"],
                pixel_size_um=pixel_size if pixel_size > 0 else None,
            )

            n_para = len(np.unique(para_labels)) - 1
            self.progress.emit(f"Stage H2 done: {n_para} parasites in hosts.")
            self.finished.emit(para_labels, vacuole_map, features, para_measurements)
        except Exception:
            import traceback

            self.error.emit(traceback.format_exc())
```

- [ ] **Step 2: Replace `_run_host_stage2`** :

```python
    def _run_host_stage2(self) -> None:
        if self._host_labels is None or self._host_labels.max() == 0:
            self._log_msg("Segment and review host cells first.")
            return
        try:
            image = self._get_image_array()
        except RuntimeError as exc:
            self._log_msg(f"Error: {exc}")
            return

        n_ch = image.shape[-1]
        ch_names = {i: f"ch{i}" for i in range(n_ch)}
        ch_names[self._ch_cptsa.value()] = "cptsa"
        ch_names[self._ch_mcherry.value()] = "mcherry"

        # Parasite/vacuole gates come from the Setup tab (PV scale), NOT the
        # host-area gates — Stage H2 is the existing PV pipeline (spec §4).
        min_area_px, max_area_px = self._pixel_area_limits()
        diameter = self._diameter.value()

        params = {
            "image": image,
            "host_labels": self._host_labels,
            "seg_ch": self._seg_ch.value(),
            "seg_backend": self._seg_backend.currentIndex(),
            "annot_dir": self._annot_dir.text(),
            "ch_cptsa": self._ch_cptsa.value(),
            "ch_mcherry": self._ch_mcherry.value(),
            "ch_names": ch_names,
            "pixel_size": self._pixel_size.value(),
            "vac_min_area_px": min_area_px,
            "vac_max_area_px": max_area_px,
            "min_area_px": min_area_px,
            "max_area_px": max_area_px,
            "max_eccentricity": self._max_eccentricity.value(),
            "min_solidity": self._min_solidity.value(),
            "use_composite": self._use_composite.isChecked(),
            "diameter": float(diameter) if diameter > 0 else None,
            "flow_threshold": self._flow_thresh.value(),
            "cellprob_threshold": self._cellprob_thresh.value(),
            "threshold_method": self._thresh_method.currentText(),
            "threshold_channel": self._thresh_channel.value(),
            "threshold_value": self._thresh_value.value(),
            "threshold_percentile": self._thresh_percentile.value(),
        }

        self._btn_host_stage2.setEnabled(False)
        self._btn_host_stage2.setText("Running…")
        self._lbl_host_stage2.setText("Detecting parasites in hosts…")

        if self._seg_backend.currentIndex() == 0:
            from ._segment import preload_model

            self._log_msg(preload_model())

        self._thread = QThread()
        self._worker = _HostParasiteWorker(params)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_host_stage2_done)
        self._worker.error.connect(self._on_host_worker_error)
        self._worker.progress.connect(self._log_msg)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.start()

    def _on_host_stage2_done(
        self,
        para_labels: np.ndarray,
        vacuole_map: dict,
        features,
        para_measurements,
    ) -> None:
        self._host_para_labels = para_labels
        self._host_vac_map = vacuole_map
        self._host_features = features
        self._host_para_measurements = para_measurements
        stem = self._image_stem

        layer_name = f"{stem}_host_parasites"
        if layer_name in self._viewer.layers:
            self._viewer.layers[layer_name].data = para_labels
        else:
            self._viewer.add_labels(para_labels, name=layer_name)

        n_para = len(np.unique(para_labels)) - 1
        self._lbl_host_stage2.setText(f"Found {n_para} parasites in hosts.")
        self._btn_host_stage2.setEnabled(True)
        self._btn_host_stage2.setText("▶ Segment parasites in hosts")
        self._btn_host_review_para.setEnabled(True)
        self._btn_host_measure.setEnabled(True)
        self._lbl_host_measure.setText("Ready — assign parasites and measure hosts.")
```

- [ ] **Step 3: Replace `_open_host_parasite_curation`** and add its save handler:

```python
    def _open_host_parasite_curation(self) -> None:
        if self._host_para_labels is None:
            self._log_msg("Run Stage H2 first.")
            return
        try:
            image = self._get_image_array()
        except RuntimeError:
            image = np.zeros((*self._host_para_labels.shape, 2), dtype=np.float32)

        n_ch = image.shape[-1]
        ch_names = {i: f"ch{i}" for i in range(n_ch)}
        ch_names[self._ch_cptsa.value()] = "cptsa"
        ch_names[self._ch_mcherry.value()] = "mcherry"
        px = self._pixel_size.value()

        from ._curation import CurationWidget

        layer_name = f"{self._image_stem}_host_parasites"
        self._host_para_curation_win = CurationWidget(
            labels=self._host_para_labels,
            image=image,
            measurements=self._host_para_measurements,
            ch_cptsa=self._ch_cptsa.value(),
            ch_mcherry=self._ch_mcherry.value(),
            ch_names=ch_names,
            pixel_size_um=px if px > 0 else None,
            vacuole_assignments=self._host_vac_map if self._host_vac_map else None,
            on_save=self._on_host_parasite_curation_saved,
            viewer=self._viewer,
            labels_layer_name=layer_name,
            parent=None,
        )
        self._host_para_curation_win.setWindowTitle(
            "Peredox — Parasite Review (Stage H2)"
        )
        self._host_para_curation_win.resize(380, 620)
        self._host_para_curation_win.show()

    def _on_host_parasite_curation_saved(
        self, decisions: dict, vacuole_assignments: dict | None = None
    ) -> None:
        """
        Persist parasite decisions from host mode.

        Parasite appearance is the same task as PV mode, so decisions feed the
        SAME curated_features.csv / RF classifier (spec §4).  StarDist training
        pairs are NOT saved from host mode: the host-masked image (zeroed
        outside hosts) is not representative of PV-mode inputs.
        """
        from ._io import append_curated_annotations
        from ._learning import train_classifier

        annot_dir = self._annot_dir.text()
        csv_path = append_curated_annotations(
            decisions=decisions,
            features=self._host_features if self._host_features is not None else {},
            image_stem=f"{self._image_stem}_host",
            annotations_dir=annot_dir,
            vacuole_assignments=vacuole_assignments,
        )
        n_dec = sum(1 for v in decisions.values() if v in (0, 1))
        self._log_msg(f"Saved {n_dec} parasite annotations → {csv_path}")

        if vacuole_assignments:
            self._host_vac_map = dict(vacuole_assignments)

        # Drop rejected parasites from the working labels
        rejected = {pid for pid, dec in decisions.items() if dec == 0}
        if rejected and self._host_para_labels is not None:
            for pid in rejected:
                self._host_para_labels[self._host_para_labels == pid] = 0
            self._host_vac_map = {
                pid: vid
                for pid, vid in self._host_vac_map.items()
                if pid not in rejected
            }
            layer_name = f"{self._image_stem}_host_parasites"
            if layer_name in self._viewer.layers:
                self._viewer.layers[layer_name].data = self._host_para_labels

        clf = train_classifier(csv_path)
        if clf is not None:
            self._classifier = clf
            self._log_msg("Classifier retrained.")
        self._update_clf_status()
        self._lbl_host_measure.setText("Parasites curated — ready to measure.")
```

- [ ] **Step 4: Ruff, import check, run suite, commit**

```bash
uv run ruff check --fix napari_peredox/ tests/ && uv run ruff format napari_peredox/ tests/
uv run python -c "from napari_peredox import _widget; print('OK')"
uv run pytest tests/ -v
git add napari_peredox/_widget.py
git commit -m "feat: add Stage H2 parasite detection inside accepted hosts"
```

---

### Task 11: Stage H3 — assign, measure, export

**Files:**
- Modify: `napari_peredox/_widget.py` — replace `_run_host_measure` and `_export_host_csv` placeholders.

**Interfaces:**
- Consumes: `assign_to_hosts()`, `measure_hosts()` (Tasks 3-4), `save_measurements()`, `save_labels()`.
- Produces: `self._host_measurements` (spec §3.1 table) and `self._host_para_measurements` gains a `host_id` column; export writes `results/<stem>_host_measurements.csv`, `<stem>_host_parasite_measurements.csv`, `<stem>_host_labels.tif`, `<stem>_host_parasite_labels.tif` via the existing `_io` helpers with stem suffixes `_host` / `_host_parasite`.

- [ ] **Step 1: Replace the two placeholders**

```python
    # ── Stage H3: assignment, measurement, export ────────────────────────────

    def _run_host_measure(self) -> None:
        if self._host_labels is None:
            self._log_msg("Run Stage H1 first.")
            return
        if self._host_para_labels is None:
            # Zero parasites is valid — an uninfected control image (spec §7)
            self._host_para_labels = np.zeros_like(self._host_labels)
            self._host_vac_map = {}
            self._log_msg("No Stage H2 parasites — measuring hosts as uninfected.")
        try:
            image = self._get_image_array()
        except RuntimeError as exc:
            self._log_msg(f"Error: {exc}")
            return

        n_ch = image.shape[-1]
        ch_names = {i: f"ch{i}" for i in range(n_ch)}
        ch_names[self._ch_cptsa.value()] = "cptsa"
        ch_names[self._ch_mcherry.value()] = "mcherry"
        px = self._pixel_size.value()

        from ._host import assign_to_hosts, measure_hosts

        para_to_host, vac_to_host, dropped = assign_to_hosts(
            self._host_para_labels,
            self._host_labels,
            self._host_vac_map if self._host_vac_map else None,
        )
        if dropped:
            self._log_msg(
                f"{len(dropped)} parasite(s) had no host majority and were "
                f"dropped from host statistics: labels {sorted(dropped)}"
            )

        hosts_df = measure_hosts(
            host_labels=self._host_labels,
            para_labels=self._host_para_labels,
            image=image,
            para_to_host=para_to_host,
            vac_to_host=vac_to_host,
            dilation_px=self._host_dilation_px.value(),
            ch_cptsa=self._ch_cptsa.value(),
            ch_mcherry=self._ch_mcherry.value(),
            ch_names=ch_names,
            pixel_size_um=px if px > 0 else None,
        )
        self._host_measurements = hosts_df

        # Tag each parasite row with its host
        if (
            self._host_para_measurements is not None
            and not self._host_para_measurements.empty
        ):
            self._host_para_measurements = self._host_para_measurements.copy()
            self._host_para_measurements["host_id"] = (
                self._host_para_measurements.index.map(para_to_host)
            )

        n_inf = int(hosts_df["infected"].sum()) if not hosts_df.empty else 0
        n_tot = len(hosts_df)
        n_empty = (
            int(hosts_df["cytosol_empty"].sum()) if not hosts_df.empty else 0
        )
        msg = f"{n_tot} hosts measured — {n_inf} infected, {n_tot - n_inf} uninfected."
        if n_empty:
            msg += f" {n_empty} host(s) fully covered by parasites (NaN ratio)."
        self._lbl_host_measure.setText(msg)
        self._log_msg(f"Stage H3 done: {msg}")
        self._btn_host_export.setEnabled(True)

        # Show the host table (reuse the PV table window pattern)
        df = hosts_df.reset_index()
        win = QWidget(self, Qt.Window)
        win.setWindowTitle("Host Measurements")
        win.resize(1000, 400)
        layout = QVBoxLayout(win)
        table = QTableWidget(len(df), len(df.columns))
        table.setHorizontalHeaderLabels([str(c) for c in df.columns])
        for row_idx, row in df.iterrows():
            for col_idx, val in enumerate(row):
                item = QTableWidgetItem(
                    f"{val:.4f}" if isinstance(val, float) else str(val)
                )
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                table.setItem(row_idx, col_idx, item)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        layout.addWidget(table)
        win.show()

    def _export_host_csv(self) -> None:
        from ._io import save_labels, save_measurements

        if self._host_measurements is None:
            self._log_msg("Run Stage H3 measurement first.")
            return
        annot_dir = self._annot_dir.text()
        stem = self._image_stem
        # Stem suffixes produce the spec §3 filenames via the existing helpers:
        # <stem>_host_measurements.csv, <stem>_host_labels.tif, etc.
        host_csv = save_measurements(self._host_measurements, f"{stem}_host", annot_dir)
        save_labels(self._host_labels, f"{stem}_host", annot_dir)
        if (
            self._host_para_measurements is not None
            and not self._host_para_measurements.empty
        ):
            save_measurements(
                self._host_para_measurements, f"{stem}_host_parasite", annot_dir
            )
        if self._host_para_labels is not None and self._host_para_labels.max() > 0:
            save_labels(self._host_para_labels, f"{stem}_host_parasite", annot_dir)
        self._log_msg(f"Host results saved → {host_csv}")
```

- [ ] **Step 2: Ruff, import check, run suite, commit**

```bash
uv run ruff check --fix napari_peredox/ tests/ && uv run ruff format napari_peredox/ tests/
uv run python -c "from napari_peredox import _widget; print('OK')"
uv run pytest tests/ -v
git add napari_peredox/_widget.py
git commit -m "feat: add Stage H3 host assignment, measurement, and export"
```

---

### Task 12: batch host mode — worker path

**Files:**
- Modify: `napari_peredox/_batch.py` — `_BatchWorker.run()` (line ~379) gains a host-mode branch; `BatchWidget._run()` (line ~1331) passes the new params.
- Test: worker logic is thin orchestration over already-tested `_host.py` functions; verified by import check + the Task 13 UI walkthrough. No new unit tests.

**Interfaces:**
- Consumes: `segment_host_cells`, `assign_to_hosts`, `measure_hosts`, `segment_pvs`, `segment_parasites_in_vacuoles`, `apply_classifier_filter`, `extract_features`, `measure_pvs`, `load_classifier(..., filename="curated_host_features.joblib")`.
- Produces: in host mode, `finished` emits `(result_obj, curation_list)` where `result_obj` is a dict `{key: {"hosts": DataFrame, "parasites": DataFrame}}` keyed by `(file_stem, pos_name)`, and each `curation_list` entry additionally carries `"mode": "host"`, `"host_labels"`, `"para_labels"`, `"para_to_host"`, `"vac_map"`. PV mode emits exactly what it does today.
- New params read by the worker: `analysis_mode` (`"pv"`/`"host"`), `host_ch`, `clip_percentile`, `host_min_area_um2`, `host_max_area_um2`, `host_dilation_px`, `host_classifier`.

- [ ] **Step 1: Add the host branch to `_BatchWorker.run()`**

At the top of `run()`, after `source_type` is read, add:

```python
            analysis_mode: str = p.get("analysis_mode", "pv")
```

Then, inside the per-position loop (`for pos_idx, (file_stem, pos_name, image, file_px) in enumerate(work_items):`), immediately after the MIP-saving block, insert the host branch — the `continue` means the existing Stage-1 code below needs no changes:

```python
                if analysis_mode == "host":
                    self._process_host_position(
                        p, image, file_stem, pos_name, file_px,
                        pos_idx, total, mask_dir, host_results, curation_list,
                    )
                    continue
```

Before the loop, alongside `curation_list`, add:

```python
            host_results: dict = {}  # (file_stem, pos_name) -> {"hosts": df, "parasites": df}
```

and change the final emit to:

```python
            if analysis_mode == "host":
                self.finished.emit(host_results, curation_list)
            else:
                self.finished.emit(pd.DataFrame(), curation_list)
```

- [ ] **Step 2: Add `_process_host_position` as a method of `_BatchWorker`**

```python
    def _process_host_position(
        self, p, image, file_stem, pos_name, file_px,
        pos_idx, total, mask_dir, host_results, curation_list,
    ):
        """Full auto host pipeline for one position: H1 → H2 → H3 (spec §5)."""
        from ._host import assign_to_hosts, measure_hosts, segment_host_cells
        from ._measure import measure_pvs
        from ._segment import segment_parasites_in_vacuoles, segment_pvs

        safe_pos = _safe_filename(pos_name)
        ch_cptsa = p.get("ch_cptsa", 0)
        ch_mcherry = p.get("ch_mcherry", 1)
        n_ch = image.shape[-1]
        ch_names = {i: f"ch{i}" for i in range(n_ch)}
        ch_names[ch_cptsa] = "cptsa"
        ch_names[ch_mcherry] = "mcherry"

        if file_px > 0:
            host_min_px = p["host_min_area_um2"] / (file_px**2)
            host_max_px = p["host_max_area_um2"] / (file_px**2)
            vac_min_px = p.get("vac_min_area_um2", 20.0) / (file_px**2)
            vac_max_px = p.get("vac_max_area_um2", 2000.0) / (file_px**2)
        else:
            host_min_px, host_max_px = 0.0, 1e9
            vac_min_px, vac_max_px = 0.0, 1e9

        # ── Stage H1: hosts (+ optional host classifier) ─────────────────────
        host_labels, _, hstats = segment_host_cells(
            image=image,
            channel_index=p.get("host_ch", ch_mcherry),
            clip_percentile=p.get("clip_percentile", 99.0),
            diameter=p.get("diameter"),
            flow_threshold=p.get("flow_threshold", 0.4),
            cellprob_threshold=p.get("cellprob_threshold", 0.0),
            min_area_px=host_min_px,
            max_area_px=host_max_px,
        )
        host_clf = p.get("host_classifier")
        if host_clf is not None and host_labels.max() > 0:
            from ._learning import extract_features
            from ._segment import apply_classifier_filter

            feats = extract_features(
                labels=host_labels, image=image,
                seg_channel=p.get("host_ch", ch_mcherry),
                ch_cptsa=ch_cptsa, ch_mcherry=ch_mcherry, ch_names=ch_names,
            )
            host_labels = apply_classifier_filter(host_labels, feats, host_clf)
        n_hosts = len(np.unique(host_labels)) - 1
        self.progress.emit(
            pos_idx, total,
            f"    Hosts: {hstats['total_raw']} raw → {n_hosts} kept.",
        )
        if n_hosts == 0:
            self.progress.emit(pos_idx, total, "    No hosts — position skipped.")
            return

        # ── Stage H2: parasites inside hosts (auto) ──────────────────────────
        masked = image * (host_labels > 0)[..., np.newaxis].astype(image.dtype)
        vac_labels, _, _ = segment_pvs(
            image=masked,
            channel_index=p.get("seg_ch", 0),
            use_composite=p.get("use_composite", False),
            min_area_px=vac_min_px,
            max_area_px=vac_max_px,
            max_eccentricity=p.get("max_eccentricity", 0.85),
            min_solidity=p.get("min_solidity", 0.70),
            diameter=None,
            flow_threshold=p.get("flow_threshold", 0.4),
            cellprob_threshold=p.get("cellprob_threshold", 0.0),
            threshold_method=p.get("threshold_method", "none"),
            threshold_channel=p.get("threshold_channel", 0),
            threshold_value=p.get("threshold_value", 0.0),
            threshold_percentile=p.get("threshold_percentile", 50.0),
        )
        para_labels, vac_map = segment_parasites_in_vacuoles(
            image=masked,
            vac_labels=vac_labels,
            seg_channel=p.get("seg_ch", 0),
            model=None,
            min_area_px=p.get("min_area_um2", 5.0) / (file_px**2) if file_px > 0 else 0.0,
            max_area_px=p.get("max_area_um2", 200.0) / (file_px**2) if file_px > 0 else 1e9,
            max_eccentricity=p.get("max_eccentricity", 0.85),
            min_solidity=p.get("min_solidity", 0.70),
        )

        # ── Stage H3: assign + measure ───────────────────────────────────────
        para_to_host, vac_to_host, dropped = assign_to_hosts(
            para_labels, host_labels, vac_map
        )
        if dropped:
            self.progress.emit(
                pos_idx, total,
                f"    {len(dropped)} parasite(s) without host majority dropped.",
            )
        hosts_df = measure_hosts(
            host_labels=host_labels,
            para_labels=para_labels,
            image=image,
            para_to_host=para_to_host,
            vac_to_host=vac_to_host,
            dilation_px=p.get("host_dilation_px", 3),
            ch_cptsa=ch_cptsa,
            ch_mcherry=ch_mcherry,
            ch_names=ch_names,
            pixel_size_um=file_px if file_px > 0 else None,
        )
        para_df = measure_pvs(
            labels=para_labels, image=image,
            ch_cptsa=ch_cptsa, ch_mcherry=ch_mcherry, ch_names=ch_names,
            pixel_size_um=file_px if file_px > 0 else None,
        )
        if not para_df.empty:
            para_df["host_id"] = para_df.index.map(para_to_host)

        # Experimental metadata (same tagging idea as PV mode)
        for df in (hosts_df, para_df):
            if df is not None and not df.empty:
                df["file"] = file_stem
                df["position"] = pos_name
                df["treatment"] = p["treatment"]
                df["cell_line"] = p["cell_line"]
                df["replicate"] = p["replicate"]

        n_inf = int(hosts_df["infected"].sum()) if not hosts_df.empty else 0
        self.progress.emit(
            pos_idx, total,
            f"    {len(hosts_df)} hosts ({n_inf} infected), "
            f"{len(para_df)} parasites.",
        )

        # ── Persist masks; stash results + curation entry ────────────────────
        try:
            save_mask_tiff(host_labels, mask_dir / f"{file_stem}_{safe_pos}_host_mask.tif")
            save_mask_tiff(
                para_labels, mask_dir / f"{file_stem}_{safe_pos}_host_para_mask.tif"
            )
        except Exception as exc:
            self.progress.emit(pos_idx, total, f"    WARNING: mask save failed: {exc}")

        key = (file_stem, pos_name)
        host_results[key] = {"hosts": hosts_df, "parasites": para_df}
        curation_list.append(
            {
                "mode": "host",
                "display_name": f"{file_stem} | {pos_name}",
                "file": file_stem,
                "position_name": pos_name,
                "image": image,
                "host_labels": host_labels,
                "para_labels": para_labels,
                "para_to_host": para_to_host,
                "vac_map": vac_map,
                "file_px": file_px,
                "treatment": p["treatment"],
                "cell_line": p["cell_line"],
                "replicate": p["replicate"],
                "pos_idx": pos_idx,
            }
        )
```

- [ ] **Step 3: Pass the new params from `BatchWidget._run()`**

In the `params = {...}` dict add (widget attrs arrive in Task 13; guard with `getattr` is NOT needed because Task 13 lands before this is reachable — but to keep this task independently committable, read them defensively):

```python
            "analysis_mode": (
                "host"
                if getattr(self, "_analysis_mode", None) is not None
                and self._analysis_mode.currentIndex() == 1
                else "pv"
            ),
            "host_ch": getattr(self, "_host_ch", None).value() if getattr(self, "_host_ch", None) else 1,
            "clip_percentile": getattr(self, "_host_clip_pct", None).value() if getattr(self, "_host_clip_pct", None) else 99.0,
            "host_min_area_um2": getattr(self, "_host_min_area_um2", None).value() if getattr(self, "_host_min_area_um2", None) else 200.0,
            "host_max_area_um2": getattr(self, "_host_max_area_um2", None).value() if getattr(self, "_host_max_area_um2", None) else 10000.0,
            "host_dilation_px": getattr(self, "_host_dilation_px", None).value() if getattr(self, "_host_dilation_px", None) else 3,
            "host_classifier": self._load_host_classifier_if_requested(),
```

and add the helper method to `BatchWidget`:

```python
    def _load_host_classifier_if_requested(self):
        """Host classifier for batch host mode; None when unavailable/not requested."""
        if not self._use_classifier.isChecked():
            return None
        from ._learning import load_classifier

        clf = load_classifier(
            self._annot_dir.text(), filename="curated_host_features.joblib"
        )
        if clf is None:
            self._log_msg("No host classifier trained yet — batch runs without it.")
        return clf
```

- [ ] **Step 4: Ruff, import check, commit**

```bash
uv run ruff check --fix napari_peredox/ tests/ && uv run ruff format napari_peredox/ tests/
uv run python -c "from napari_peredox import _batch; print('OK')"
uv run pytest tests/ -v
git add napari_peredox/_batch.py
git commit -m "feat: add host-mode pipeline to batch worker"
```

---

### Task 13: batch host mode — UI, review, save

**Files:**
- Modify: `napari_peredox/_batch.py` — `_build_source_tab` (mode dropdown), `_build_channels_tab` (host params group), `_on_finished` (host branch), `_open_curation` (host branch), `_save_accepted_results` (host branch), `__init__` (`self._host_results = {}`).

**Interfaces:**
- Consumes: Task 12's `finished` payload; `VacuoleCurationWidget(object_name="host cell")`; `assign_to_hosts` + `measure_hosts` for post-curation recompute.
- Produces: "Save accepted results" in host mode writes `<out_folder>/hosts.csv` and `<out_folder>/host_parasites.csv` (concatenation of per-position frames, rejected hosts and their parasites removed).

- [ ] **Step 1: Mode dropdown in `_build_source_tab`** — at the top of the tab layout add:

```python
        mode_box = QGroupBox("Analysis mode")
        mode_layout = QVBoxLayout(mode_box)
        self._analysis_mode = QComboBox()
        self._analysis_mode.addItems(["Vacuoles / PVs", "Host cells"])
        self._analysis_mode.setToolTip(
            "Vacuoles/PVs — existing two-stage PV pipeline.\n"
            "Host cells — segment Peredox-expressing hosts, then parasites\n"
            "inside them; outputs hosts.csv + host_parasites.csv."
        )
        mode_layout.addWidget(self._analysis_mode)
        layout.insertWidget(0, mode_box)
```

(Adapt `layout.insertWidget(0, …)` to the tab's actual root layout variable name.)

- [ ] **Step 2: Host params group in `_build_channels_tab`** — append a group enabled only in host mode:

```python
        host_box = QGroupBox("Host mode settings")
        host_form = QFormLayout(host_box)
        host_form.setContentsMargins(6, 6, 6, 6)

        self._host_ch = QSpinBox()
        self._host_ch.setRange(0, 15)
        self._host_ch.setValue(1)
        host_form.addRow("Host channel:", self._host_ch)

        self._host_clip_pct = QDoubleSpinBox()
        self._host_clip_pct.setRange(50.0, 100.0)
        self._host_clip_pct.setDecimals(1)
        self._host_clip_pct.setValue(99.0)
        host_form.addRow("Clip percentile:", self._host_clip_pct)

        self._host_min_area_um2 = QDoubleSpinBox()
        self._host_min_area_um2.setRange(0.0, 1e6)
        self._host_min_area_um2.setDecimals(0)
        self._host_min_area_um2.setValue(200.0)
        self._host_max_area_um2 = QDoubleSpinBox()
        self._host_max_area_um2.setRange(0.0, 1e6)
        self._host_max_area_um2.setDecimals(0)
        self._host_max_area_um2.setValue(10000.0)
        host_area_row = QHBoxLayout()
        host_area_row.addWidget(QLabel("min:"))
        host_area_row.addWidget(self._host_min_area_um2)
        host_area_row.addWidget(QLabel("max:"))
        host_area_row.addWidget(self._host_max_area_um2)
        host_form.addRow("Host area (µm²):", host_area_row)

        self._host_dilation_px = QSpinBox()
        self._host_dilation_px.setRange(0, 50)
        self._host_dilation_px.setValue(3)
        host_form.addRow("Parasite exclusion buffer (px):", self._host_dilation_px)

        host_box.setEnabled(False)
        self._analysis_mode.currentIndexChanged.connect(
            lambda i: host_box.setEnabled(i == 1)
        )
        layout.addWidget(host_box)
```

- [ ] **Step 3: `_on_finished` host branch** — at the top of `_on_finished`:

```python
        if curation_list and curation_list[0].get("mode") == "host":
            self._progress_bar.setValue(100)
            self._host_results = result  # dict {(file, pos) -> {"hosts", "parasites"}}
            self._result_df = None
            self._curation_decisions = {}
            self._curation_data = curation_list
            self._curation_combo.clear()
            for item in curation_list:
                self._curation_combo.addItem(item["display_name"])
            self._curation_combo.setEnabled(True)
            self._btn_curate.setEnabled(True)
            self._btn_save_results.setEnabled(True)
            self._update_review_status()
            n_hosts = sum(len(v["hosts"]) for v in result.values())
            self._log_msg(
                f"Host batch complete — {len(curation_list)} position(s), "
                f"{n_hosts} host cells measured.\n"
                f"Review positions to reject bad host masks, then save."
            )
            return
```

Also initialize `self._host_results = {}` in `BatchWidget.__init__` next to `self._result_df`.

- [ ] **Step 4: `_open_curation` host branch** — at the top of `_open_curation`, before the existing two-pass flow:

```python
        idx = self._curation_combo.currentIndex()
        if idx < 0 or idx >= len(self._curation_data):
            return
        item = self._curation_data[idx]
        if item.get("mode") == "host":
            self._open_host_position_curation(item)
            return
```

(then let the existing code re-fetch `idx`/`item` as it already does), and add:

```python
    def _open_host_position_curation(self, item: dict):
        """Accept/reject/redraw host masks for one batch position."""
        from ._curation import VacuoleCurationWidget

        def _on_save(decisions: dict, curated_hosts: np.ndarray):
            from ._host import assign_to_hosts, measure_hosts

            hosts = curated_hosts.copy()
            for hid, dec in decisions.items():
                if dec == 0:
                    hosts[hosts == hid] = 0
            item["host_labels"] = hosts

            # Recompute assignment + measurement against the curated hosts.
            # Parasites in rejected hosts lose their majority and are dropped.
            para_to_host, vac_to_host, _dropped = assign_to_hosts(
                item["para_labels"], hosts, item.get("vac_map") or None
            )
            ch_names = {0: "ch0", 1: "ch1"}
            ch_names[self._ch_cptsa.value()] = "cptsa"
            ch_names[self._ch_mcherry.value()] = "mcherry"
            file_px = item["file_px"]
            hosts_df = measure_hosts(
                host_labels=hosts,
                para_labels=item["para_labels"],
                image=item["image"],
                para_to_host=para_to_host,
                vac_to_host=vac_to_host,
                dilation_px=self._host_dilation_px.value(),
                ch_cptsa=self._ch_cptsa.value(),
                ch_mcherry=self._ch_mcherry.value(),
                ch_names=ch_names,
                pixel_size_um=file_px if file_px > 0 else None,
            )
            from ._measure import measure_pvs

            para_df = measure_pvs(
                labels=item["para_labels"],
                image=item["image"],
                ch_cptsa=self._ch_cptsa.value(),
                ch_mcherry=self._ch_mcherry.value(),
                ch_names=ch_names,
                pixel_size_um=file_px if file_px > 0 else None,
            )
            if not para_df.empty:
                para_df["host_id"] = para_df.index.map(para_to_host)
                para_df = para_df[para_df["host_id"].notna()]
            for df in (hosts_df, para_df):
                if df is not None and not df.empty:
                    df["file"] = item["file"]
                    df["position"] = item["position_name"]
                    df["treatment"] = item["treatment"]
                    df["cell_line"] = item["cell_line"]
                    df["replicate"] = item["replicate"]

            key = (item["file"], item["position_name"])
            self._host_results[key] = {"hosts": hosts_df, "parasites": para_df}
            self._curation_decisions[key] = dict(decisions)
            self._update_review_status()
            self._log_msg(
                f"Host review saved for {item['display_name']} — "
                f"{len(hosts_df)} hosts kept."
            )

        self._host_curation_win = VacuoleCurationWidget(
            vac_labels=item["host_labels"],
            image=item["image"],
            ch_cptsa=self._ch_cptsa.value(),
            ch_mcherry=self._ch_mcherry.value(),
            on_save=_on_save,
            viewer=self._viewer,
            labels_layer_name=None,
            object_name="host cell",
            parent=None,
        )
        self._host_curation_win.setWindowTitle(
            f"Host Review — {item['display_name']}"
        )
        self._host_curation_win.resize(360, 560)
        self._host_curation_win.show()
```

Note: `BatchWidget` may have `self._viewer = napari_viewer` already; if the attribute is named differently, pass whatever the existing `_open_curation` passes for its viewer argument.

- [ ] **Step 5: `_save_accepted_results` host branch** — at the top:

```python
        if getattr(self, "_host_results", None):
            out_folder = self._pending_out_folder
            out_folder.mkdir(parents=True, exist_ok=True)
            hosts_all = pd.concat(
                [v["hosts"] for v in self._host_results.values() if not v["hosts"].empty]
            )
            paras = [
                v["parasites"]
                for v in self._host_results.values()
                if v["parasites"] is not None and not v["parasites"].empty
            ]
            hosts_path = out_folder / "hosts.csv"
            hosts_all.to_csv(hosts_path)
            self._log_msg(f"Saved {len(hosts_all)} host row(s) → {hosts_path}")
            if paras:
                paras_all = pd.concat(paras)
                paras_path = out_folder / "host_parasites.csv"
                paras_all.to_csv(paras_path)
                self._log_msg(
                    f"Saved {len(paras_all)} parasite row(s) → {paras_path}"
                )
            return
```

- [ ] **Step 6: Ruff, import check, run suite, commit**

```bash
uv run ruff check --fix napari_peredox/ tests/ && uv run ruff format napari_peredox/ tests/
uv run python -c "from napari_peredox import _batch; print('OK')"
uv run pytest tests/ -v
git add napari_peredox/_batch.py
git commit -m "feat: add host-mode UI, review, and save to batch widget"
```

---

### Task 14: documentation + final verification

**Files:**
- Modify: `CLAUDE.md` (architecture section, workflow notes)

**Interfaces:** none — documentation only.

- [ ] **Step 1: Update `CLAUDE.md`**

Add `_host.py` to the Architecture diagram/table, a "Host analysis workflow" subsection describing: Host tab stages H1→review→H2→review→H3, batch host mode, the population rule (all accepted hosts measured infected or not; parasites only inside accepted hosts), output filenames (spec §3), and host classifier files (`curated_host_features.csv/.joblib`). Add a "Tests" section: `uv run pytest tests/ -v`. Update the "State of the project" date stamp.

- [ ] **Step 2: Full verification**

```bash
uv run pytest tests/ -v
uv run ruff check napari_peredox/ tests/
uv run python -c "from napari_peredox import _widget, _batch, _curation, _host; print('all OK')"
```

Expected: all tests pass, ruff clean, imports OK.

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: document host-cell analysis mode"
```

- [ ] **Step 4: Manual acceptance (user)**

Launch napari from the project venv, open a U2OS Peredox image, and run the Host tab end-to-end: H1 segmentation quality, host review, H2 parasite detection, H3 table (`infected` split, ratio sanity), CSV export. This is the spec §8 acceptance test — cpSAM behavior on real data cannot be unit-tested.
