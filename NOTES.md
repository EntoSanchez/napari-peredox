# napari-peredox — status notes

Repo-level status: what changed recently and what still needs work.
Script-specific notes live in [`scripts/NOTES.md`](scripts/NOTES.md); the
architecture reference is [`CLAUDE.md`](CLAUDE.md).

**As of 2026-10-07** — pushed through `527222f`. 52 tests passing, ruff clean,
all modules import. The host-cell analysis mode is feature-complete in code but
**the batch three-pass curation chain has not yet been run inside napari**.

---

## Recent changes (2026-08-25 → 2026-09-02)

### 1. Host-cell analysis mode (new)

Second analysis pipeline alongside the original PV one: segment U2OS host cells
expressing cytosolic Peredox, find parasites inside them, and report per-host
Peredox ratio with infection status. Design:
[`docs/superpowers/specs/2026-08-25-host-analysis-design.md`](docs/superpowers/specs/2026-08-25-host-analysis-design.md).

- New module `_host.py`: `clip_bright`, `segment_host_cells`, `assign_to_hosts`,
  `assign_vacuoles_to_hosts`, `measure_hosts`, `measure_vacuoles_in_hosts`,
  `host_vacuole_summary`, `vacuoles_from_parasites`, `drop_infection_columns`.
- Single-image **Host tab** (4th tab) and batch host modes.
- Host classifier kept **entirely separate** from the PV one
  (`curated_host_features.csv` / `.joblib`); batch host review now feeds and
  retrains it, and its status is shown in both widgets.

### 2. Three analysis modes

Selector moved to the **top of the batch Channels tab**, with only the relevant
setting groups visible per mode:

| Mode | Hosts | Parasites |
|---|---|---|
| Parasites / PVs | — | original two-stage pipeline |
| Host cells + parasites | segmented + curated | detected inside accepted hosts |
| Host cells only | segmented + curated | never looked for |

Host-only exports **drop** the infection columns (`_host.INFECTION_COLS`) so
"never checked" can't be misread as "verified uninfected".

### 3. Curation order inverted — hosts first, vacuoles last

The batch worker now runs **Stage H1 only**; each position is reviewed as a
three-pass chain, each stage detected inside what the previous one accepted:

1. **hosts** → 2. **parasites** (inside accepted hosts) → 3. **vacuoles**
(grouped from accepted parasites; rejecting one drops its parasites)

Measurement runs after pass 3, so saved rows describe exactly what was accepted,
and curated masks are written back.

**Vacuoles are never segmented standalone.** cpSAM on a host-masked image
returns host **nuclei** — once masked to hosts they're the roundest
high-contrast objects, and at ~150–250 µm² they pass the 20–2000 µm² gate.
Measured: only ~half of parasites fell inside a detected "vacuole" and two
thirds of detections held no parasite. Parasites are unambiguous, so they go
first and vacuoles are recovered by dilate → connect → fill
(`_host.vacuoles_from_parasites`).

Fine-tuned **StarDist models are now wired into host mode** (previously
hardcoded `model=None` with a "not supported in batch host mode" notice).

### 4. Vacuole reporting folded into `hosts.csv`

Per your call, no separate vacuole table — `hosts.csv` gained
`mean_parasites_per_vacuole`, `max_parasites_per_vacuole`,
`vacuole_area_px_total`, `vacuole_area_um2_total`, `mean_vacuole_area_px`,
`mean_vacuole_ratio_intden`, `median_vacuole_ratio_intden`,
`mean_parasite_ratio`, `median_parasite_ratio`, `n_vacuoles_with_parasites`,
`excluded_area_px`. `host_parasites.csv` gained `vacuole_id`.

Two correctness fixes alongside:
- **Host cytosol now excludes the whole vacuole mask**, not just parasite
  bodies — mCherry fills the PV lumen, so inter-parasite space was leaking
  parasite signal into host ratios.
- **`n_vacuoles` counts from the vacuole mask directly.** It previously saw
  vacuoles only through their member parasites, so a vacuole whose parasites
  weren't resolved went uncounted.

### 5. Re-measure without re-segmenting

- Batch checkbox **"Re-measure: reuse saved masks (skip segmentation)"**.
- [`scripts/remeasure_host_run.py`](scripts/remeasure_host_run.py) — scriptable
  equivalent, with `--curated-from` to preserve review decisions (saved masks
  are written *before* curation, so a naive re-measure resurrects rejected
  hosts — it silently re-added 8 and 5 hosts in the two KO arms).
- [`scripts/backfill_vacuole_masks.py`](scripts/backfill_vacuole_masks.py) —
  adds missing vacuole masks to old runs (`--from-parasites` is the mode to
  use).
- [`scripts/check_vacuole_masks.py`](scripts/check_vacuole_masks.py) — QC:
  what fraction of parasites sit inside a vacuole, how many vacuoles are empty.
  **Always run after a backfill.**

### 6. Tuned segmentation defaults (from a real-data sweep)

[`scripts/host_param_sweep.py`](scripts/host_param_sweep.py) established that
**cpSAM auto-diameter is unusable** on dim Peredox images (it shatters cells
into ~1 µm² speckle — this was the "only segmenting nuclei" report). Defaults
now: host diameter **300 px**, min host area **350 µm²** (above a U2OS nucleus),
clip percentile 99.

### 7. Downstream analysis

[`D:\Lourido Lab\figures\UZS01_range_peredox\`](../figures/UZS01_range_peredox/)
— own uv project: master + seeded random-50-per-replicate datasets, ANOVA/Tukey
at **both** cell and replicate-mean level, violin+beeswarm figures in three
themes, plus a by-replicate variant. Result: pyruvate lowers the ratio ~0.23
(robust at both levels); lactate and oligomycin are indistinguishable from
vehicle.

---

## Needs improvement

Roughly in priority order.

### High — correctness / blocking

1. **The three-pass curation chain has never been run in napari.** Ruff, imports
   and 52 tests pass, but none of them exercise Qt callbacks. An adversarial
   review was started and interrupted. Most-likely failure points: curation
   widget callback signatures, closure capture in the pass-to-pass handoffs,
   and widget garbage collection mid-chain. **Trial-run one position before a
   real batch.**

2. **The PV RandomForest classifier is inert.** `self._classifier` is loaded,
   trained and displayed, but `apply_classifier_filter` is only ever called for
   *hosts*. In `_widget.py` the `_use_classifier` checkbox is created and never
   read; in `_batch.py` it's put into params (line ~1973) and never consumed.
   The 85 MB `curated_features.joblib` has never filtered anything. Either wire
   it up or remove the checkboxes — right now they silently lie.

3. **The 20260824 infection data needs re-running through the new flow.** Its
   parasite masks came from the old vacuole-first order and badly
   under-detected: on re-test, pos009 went 10 → **40** parasites and pos020
   0 → **18**. The infection counts in the current CSVs (18 and 22 hosts) are
   almost certainly too low. Channels and curation in those files are correct;
   the parasite detection is not.

### Medium — scientific interpretation

4. **The vacuole "lumen" is an approximation.** `vacuoles_from_parasites` gives
   a tight envelope around the rosette (dilate 5 px + fill), so
   `mean_vacuole_ratio_intden` is a rosette-plus-margin ratio, not a true
   full-lumen ratio. A genuine lumen measurement needs a PV-specific marker
   channel (GRA, or the 640 IMC1-Halo channel) rather than mCherry.

5. **Dim / low-expressing host cells are never segmented.** Visible in the
   parameter sweep — faint cells go undetected at every setting. That's a
   systematic exclusion of low expressers and may bias the ratio distribution.
   Decide whether it's intentional (unreliable signal) or needs fixing.

6. **`infected` is defined by parasites, not vacuoles.** Now that vacuoles are
   counted independently, a host can in principle show `n_vacuoles > 0` with
   `infected = False`. Harmless today (vacuoles are derived from parasites) but
   worth revisiting if vacuole detection ever becomes independent again.

7. **Unreviewed batch positions contribute no rows.** Inherent to the new order
   — detection can't precede the curation it depends on — but it means a batch
   run is only as complete as the review. The log says so on completion.

### Low — ergonomics / robustness

8. **`host_id` and `vacuole_id` are per-position, not global.** Any analysis
   must group on `(position, host_id)`. Easy to get silently wrong; consider
   emitting a globally unique key.

9. **The helper scripts assume doubled mask filenames.** Masks are named
   `{file_stem}_{safe_pos}_host_mask.tif`; for a TIFF-folder source both halves
   are identical, and `remeasure_host_run.py` / `backfill_vacuole_masks.py`
   recover the stem with `stem[:len(stem)//2]`. **That heuristic breaks for
   ND2-sourced runs**, where the two halves differ. Needs a real stem↔image
   mapping (e.g. a manifest written at run time).

10. **The worker's first host mask is still saved pre-curation.** The three-pass
    flow now writes curated masks at finish, but the initial one is not, so a
    re-measure that skips `--curated-from` can still pick up the uncurated
    version.

11. **GPU VRAM contention makes batches crawl.** The 6 GB card is shared; a
    long-running napari session holding ~3.4 GB left a backfill ~300 MB of
    headroom and slowed it dramatically (the earlier "stuck on one file"
    report). Close other GPU apps before long runs; consider logging free VRAM
    at batch start.

12. **No UI test coverage at all.** The 52 tests cover pure array/DataFrame
    logic only. A headless-Qt smoke test of the curation chain would have
    caught the ordering bug before it reached real data.

---

## Reproducibility pointers

| What | Where |
|---|---|
| Architecture + workflow reference | [`CLAUDE.md`](CLAUDE.md) |
| Host-mode design spec | [`docs/superpowers/specs/2026-08-25-host-analysis-design.md`](docs/superpowers/specs/2026-08-25-host-analysis-design.md) |
| Implementation plan | [`docs/superpowers/plans/2026-08-25-host-analysis.md`](docs/superpowers/plans/2026-08-25-host-analysis.md) |
| Script usage + real-data findings | [`scripts/NOTES.md`](scripts/NOTES.md) |
| Downstream figures | `D:\Lourido Lab\figures\UZS01_range_peredox\` |
| Tests | `uv run pytest tests/ -v` (52) |
