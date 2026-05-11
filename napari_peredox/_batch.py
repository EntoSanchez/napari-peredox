"""
_batch.py — Bulk processing of multi-position fluorescence images

Purpose
-------
Handles high-throughput batch analysis from either ND2 Z-stack files or
folders of pre-made TIFF max-intensity projections.

For every image position the pipeline:

  1. Reads the image (ND2: max-projects Z-stack; TIFF: loads directly)
  2. (ND2 mode) Saves the MIP as a multi-channel TIFF to out_folder/mips/
  3. Runs Cellpose-SAM segmentation (with optional classifier pre-filter)
  4. Saves the segmentation mask TIFF to out_folder/masks/
  5. Measures fluorescence (integrated density, mean, cpTSapphire/mCherry ratio)
  6. Appends one row per parasite to an in-memory table

After all positions are processed the combined table is written to
  out_folder/results.csv

Output folder layout
--------------------
  out_folder/
  ├── mips/            (ND2 mode only)
  │   ├── <stem>_<position>_MIP.tif
  │   └── …
  ├── masks/
  │   ├── <stem>_<position>_mask.tif
  │   └── …
  └── results.csv

Output CSV columns (one row per parasite)
------------------------------------------
  file, position_index, position_name,
  treatment, cell_line, replicate,
  parasite_label, centroid_y, centroid_x,
  area_px, [area_um2],
  mean_cptsa, intden_cptsa,
  mean_mcherry, intden_mcherry,
  ratio_cptsa_mcherry
"""

from __future__ import annotations

import re
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from qtpy.QtCore import QObject, QThread, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

# ── ND2 reader helpers ────────────────────────────────────────────────────────


def read_nd2_positions(path: str | Path) -> list[tuple[str, np.ndarray]]:
    """
    Read all stage positions from an ND2 Z-stack file.

    Each position's Z-slices are max-projected to produce a single 2-D
    multi-channel image.

    Parameters
    ----------
    path : str or Path
        Path to the .nd2 file.

    Returns
    -------
    positions : list of (position_name, image_array)
        position_name : str
            Human-readable stage position name from the microscope log, or
            'pos000', 'pos001', … if names are unavailable.
        image_array : np.ndarray, shape (H, W, C), dtype float32
            Max-intensity projection of the Z-stack for this position.
            Channels are on the last axis to match the rest of the pipeline.
    """
    import nd2

    positions = []

    with nd2.ND2File(path) as f:
        sizes = dict(f.sizes)  # e.g. {'P': 12, 'Z': 30, 'C': 2, 'Y': 512, 'X': 512}

        # ── How many stage positions? ────────────────────────────────────────
        n_positions = sizes.get("P", 1)

        # ── Get human-readable position names from the microscope metadata ───
        # Nikon NIS-Elements stores stage point names in the XYPosLoop
        # experiment descriptor.  _extract_position_names() returns a list of
        # strings (one per position), falling back to 'pos000', 'pos001', …
        pos_names = _extract_position_names(f, n_positions)

        # ── Build a lookup: position index → list of linear frame indices ────
        # nd2.ND2File.loop_indices is a list of dicts (one per frame in the
        # file), where each dict maps dimension name → integer index.
        # Example for a 3-position, 10-Z-slice, 2-channel file:
        #   frame 0:  {'P': 0, 'Z': 0, 'C': 0}
        #   frame 1:  {'P': 0, 'Z': 0, 'C': 1}
        #   frame 2:  {'P': 0, 'Z': 1, 'C': 0}  … etc.
        # We group by 'P' so we know which frames belong to each position.
        loop_idx = f.loop_indices

        pos_to_frames: dict[int, list[int]] = {p: [] for p in range(n_positions)}
        for frame_i, dims in enumerate(loop_idx):
            p = dims.get("P", 0)
            pos_to_frames[p].append(frame_i)

        # ── Read and max-project each position ───────────────────────────────
        for p_idx in range(n_positions):
            frame_indices = pos_to_frames[p_idx]
            if not frame_indices:
                continue

            # read_frame(i) returns a numpy array for one frame.
            # For a multi-channel acquisition, the shape is (C, H, W).
            # For a single-channel acquisition, the shape is (H, W).
            frames = [f.read_frame(fi) for fi in frame_indices]

            # Stack all frames for this position into a single array.
            # After np.stack(..., axis=0) the shape is:
            #   (n_frames, C, H, W)  or  (n_frames, H, W)
            # where n_frames = n_Z_slices * (number of any other loops).
            stack = np.stack(frames, axis=0).astype(np.float32)

            # Max-project along axis 0 (the frame/Z axis).
            # Result shape: (C, H, W)  or  (H, W)
            mip = stack.max(axis=0)

            # Ensure the result always has a channel axis: → (C, H, W)
            if mip.ndim == 2:
                mip = mip[np.newaxis, ...]  # single-channel: add C=1 axis

            # Convert (C, H, W) → (H, W, C) for the rest of our pipeline
            image_hwc = np.moveaxis(mip, 0, -1)

            positions.append((pos_names[p_idx], image_hwc))

    return positions


def save_mip_tiff(
    image_hwc: np.ndarray,
    out_path: Path,
) -> None:
    """
    Save a max-intensity projection as a multi-channel TIFF.

    The image is saved as (C, H, W) float32 with ImageJ-compatible metadata
    so it opens directly in Fiji with the channel slider.

    Parameters
    ----------
    image_hwc : np.ndarray, shape (H, W, C), dtype float32
        MIP image in the pipeline's native (H, W, C) format.
    out_path : Path
        Destination file path (parent directory must already exist).
    """
    import tifffile

    # Convert (H, W, C) → (C, H, W) — the standard TIFF/ImageJ axis order
    image_chw = np.moveaxis(image_hwc, -1, 0)  # (C, H, W)

    # imagej=True writes the OME-TIFF header so Fiji recognises multi-channel
    tifffile.imwrite(str(out_path), image_chw, imagej=True)


def _extract_position_names(f, n_positions: int) -> list[str]:
    """
    Pull XY stage position names out of the ND2 experiment metadata.

    Iterates through the experiment loop descriptors looking for an
    XYPosLoop whose parameters.points list contains named positions.

    Falls back to 'pos000', 'pos001', … if the metadata is missing,
    malformed, or the file format doesn't include named positions.
    """
    fallback = [f"pos{i:03d}" for i in range(n_positions)]

    try:
        for loop in f.experiment:
            if hasattr(loop, "parameters") and hasattr(loop.parameters, "points"):
                names = []
                for pt in loop.parameters.points:
                    # NIS-Elements stores the name as pt.name; older versions
                    # may not have it.  getattr with a default avoids AttributeError.
                    name = getattr(pt, "name", None)
                    names.append(str(name) if name else None)

                if len(names) == n_positions and any(n for n in names):
                    return [n or f"pos{i:03d}" for i, n in enumerate(names)]
    except Exception:
        pass  # any metadata read failure → use fallback names

    return fallback


def _safe_filename(name: str) -> str:
    """
    Convert a stage position name to a safe filename component.

    Replaces characters that are illegal or awkward in file paths
    (spaces, slashes, colons, etc.) with underscores.
    """
    return re.sub(r"[^\w\-]", "_", name).strip("_") or "pos"


def read_pixel_size_nd2(path: str | Path) -> float:
    """
    Read the physical pixel size (µm/pixel) from an ND2 file's metadata.

    Returns 0.0 if the metadata is absent or unreadable.
    """
    try:
        import nd2

        with nd2.ND2File(path) as f:
            vox = f.voxel_size()  # returns VoxelSize(x, y, z) in µm
            px = float(vox.x)
            return px if px > 0 else 0.0
    except Exception:
        return 0.0


def read_pixel_size_tiff(path: str | Path) -> float:
    """
    Read the physical pixel size (µm/pixel) from a TIFF file's tags.

    Checks XResolution / ResolutionUnit tags (ImageJ convention) and falls
    back to the OME-TIFF PhysicalSizeX field if present.
    Returns 0.0 if no calibration is found.
    """
    try:
        import tifffile

        with tifffile.TiffFile(str(path)) as tf:
            # ── OME-TIFF ───────────────────────────────────────────────────
            if tf.ome_metadata:
                import xml.etree.ElementTree as ET

                root = ET.fromstring(tf.ome_metadata)
                ns = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}
                px_el = root.find(".//ome:Pixels", ns)
                if px_el is not None:
                    size_x = px_el.get("PhysicalSizeX")
                    unit = px_el.get("PhysicalSizeXUnit", "µm")
                    if size_x:
                        val = float(size_x)
                        # Convert to µm if needed
                        if unit in ("nm", "nanometer"):
                            val /= 1000.0
                        elif unit in ("mm", "millimeter"):
                            val *= 1000.0
                        return val if val > 0 else 0.0

            # ── ImageJ / standard TIFF XResolution tag ─────────────────────
            page = tf.pages[0]
            tags = page.tags
            if 282 in tags and 296 in tags:  # XResolution, ResolutionUnit
                xres = tags[282].value  # rational: (numerator, denominator)
                unit = tags[296].value  # 1=no unit, 2=inch, 3=cm
                if isinstance(xres, tuple) and xres[1] != 0 and unit in (2, 3):
                    res_per_unit = xres[0] / xres[1]
                    if res_per_unit > 0:
                        # Convert pixels-per-unit → µm-per-pixel
                        unit_to_um = {2: 25_400.0, 3: 10_000.0}  # inch, cm
                        return unit_to_um[unit] / res_per_unit
    except Exception:
        pass
    return 0.0


def read_tiff_folder(folder: Path) -> list[tuple[str, np.ndarray]]:
    """
    Read all TIFF files in a folder as individual positions.

    Each TIFF is assumed to be a max-intensity projection (already processed).
    The file stem is used as the position name.

    Handles:
      - (H, W)       — single-channel, gains a dummy C=1 axis
      - (C, H, W)    — ImageJ/Fiji convention, converted to (H, W, C)
      - (H, W, C)    — already in pipeline format

    Parameters
    ----------
    folder : Path
        Directory containing .tif / .tiff files.

    Returns
    -------
    positions : list of (name, image_hwc)
        One entry per TIFF file, sorted alphabetically by filename.
    """
    import tifffile

    tiff_paths = sorted(
        p for p in folder.iterdir() if p.suffix.lower() in {".tif", ".tiff"}
    )
    positions = []
    for tp in tiff_paths:
        img = tifffile.imread(str(tp)).astype(np.float32)
        if img.ndim == 2:
            # (H, W) → (H, W, 1)
            img = img[..., np.newaxis]
        elif img.ndim == 3:
            # Determine axis order: if first dim is much smaller than the last
            # two, assume (C, H, W) and convert; otherwise assume (H, W, C).
            if img.shape[0] <= 16 and img.shape[1] > 16 and img.shape[2] > 16:
                img = np.moveaxis(img, 0, -1)  # (C, H, W) → (H, W, C)
            # else already (H, W, C)
        positions.append((tp.stem, img))
    return positions


def save_mask_tiff(labels: np.ndarray, out_path: Path) -> None:
    """
    Save a segmentation label array as a single-channel integer TIFF.

    Parameters
    ----------
    labels : np.ndarray (H, W) int32
        Label image — 0 = background, positive integers = parasite IDs.
    out_path : Path
        Destination file path.
    """
    import tifffile

    tifffile.imwrite(str(out_path), labels.astype(np.int32))


# ── Background worker ─────────────────────────────────────────────────────────


class _BatchWorker(QObject):
    """
    Runs the full batch pipeline in a QThread so the napari UI stays responsive.

    Signals
    -------
    progress(current, total, message)
        Emitted after each stage position is processed.
        current and total are counts of positions (not files) for the progress bar.
    finished(result_df)
        Emitted once with the combined pd.DataFrame when all files are done.
    error(traceback_str)
        Emitted if an unrecoverable exception occurs in the worker thread.
    """

    progress = Signal(int, int, str)  # current, total, message
    finished = Signal(object, object)  # pd.DataFrame, list[curation_dict]
    error = Signal(str)

    def __init__(self, params: dict):
        super().__init__()
        self.params = params

    def run(self):
        """
        Main processing loop — called by QThread.started.

        Supports two source modes:
          - 'nd2'         : reads multi-position ND2 Z-stacks, max-projects each position
          - 'tiff_folder' : reads pre-made MIP TIFFs from a folder (one file = one position)

        For each position:
          1. (ND2 mode) Max-project Z-stack → (H, W, C) float32 and save MIP TIFF
          2. Segment parasites with Cellpose-SAM (+ optional classifier pre-filter)
          3. Save segmentation mask TIFF to out_folder/masks/
          4. Measure fluorescence per parasite; optionally select one per vacuole
          5. Tag rows with experimental metadata

        Emits finished(result_df, curation_list) when done.
        curation_list contains one dict per position with image+labels+measurements
        so the user can open any position in the curation gallery.
        """
        try:
            p = self.params
            source_type: str = p.get("source_type", "nd2")
            treatment: str = p["treatment"]
            cell_line: str = p["cell_line"]
            replicate: int = p["replicate"]
            ch_cptsa: int = p["ch_cptsa"]
            ch_mcherry: int = p["ch_mcherry"]
            seg_ch: int = p["seg_ch"]
            use_composite: bool = p["use_composite"]
            min_area_um2: float = p["min_area_um2"]
            max_area_um2: float = p["max_area_um2"]
            max_eccentricity: float = p.get("max_eccentricity", 0.85)
            min_solidity: float = p.get("min_solidity", 0.70)
            pixel_size: float = p["pixel_size"]
            classifier = p["classifier"]
            out_folder: Path = p["out_folder"]
            diameter = p.get("diameter")
            flow_threshold: float = p.get("flow_threshold", 0.4)
            cellprob_threshold: float = p.get("cellprob_threshold", 0.0)

            ch_names = {ch_cptsa: "cptsa", ch_mcherry: "mcherry"}

            from ._learning import extract_features
            from ._measure import measure_pvs, select_one_per_vacuole
            from ._segment import apply_classifier_filter, group_by_vacuole, segment_pvs

            # Create output subdirectories
            mip_dir = out_folder / "mips"
            mask_dir = out_folder / "masks"
            mip_dir.mkdir(parents=True, exist_ok=True)
            mask_dir.mkdir(parents=True, exist_ok=True)

            all_rows: list[pd.DataFrame] = []
            curation_list: list[dict] = []  # per-position data for curation gallery

            # ── Build the list of (file_stem, pos_name, image, px_um) ─────────
            # pixel_size from UI takes priority; 0 = try to read from metadata.
            manual_px = pixel_size  # float; 0 means "auto-detect"
            work_items: list[tuple[str, str, np.ndarray, float]] = []

            if source_type == "nd2":
                nd2_paths: list[Path] = p["nd2_paths"]
                total_estimate = len(nd2_paths)
                processed = 0
                for nd2_path in nd2_paths:
                    file_stem = nd2_path.stem
                    self.progress.emit(
                        processed, total_estimate, f"Opening {nd2_path.name}…"
                    )
                    # Auto-detect pixel size from ND2 metadata if not set manually
                    file_px = (
                        manual_px if manual_px > 0 else read_pixel_size_nd2(nd2_path)
                    )
                    if file_px > 0 and manual_px == 0:
                        self.progress.emit(
                            processed,
                            total_estimate,
                            f"  Pixel size from metadata: {file_px:.4f} µm/px",
                        )
                    try:
                        positions = read_nd2_positions(nd2_path)
                    except Exception as exc:
                        self.progress.emit(
                            processed,
                            total_estimate,
                            f"  ERROR reading {nd2_path.name}: {exc}",
                        )
                        continue
                    total_estimate = total_estimate - 1 + len(positions)
                    for pos_name, image in positions:
                        work_items.append((file_stem, pos_name, image, file_px))
            else:
                tiff_folder = Path(p["tiff_folder"])
                self.progress.emit(0, 1, f"Scanning {tiff_folder.name}…")
                try:
                    tiff_positions = read_tiff_folder(tiff_folder)
                except Exception as exc:
                    self.error.emit(f"Could not read TIFF folder: {exc}")
                    return
                tiff_paths = sorted(
                    pp
                    for pp in tiff_folder.iterdir()
                    if pp.suffix.lower() in {".tif", ".tiff"}
                )
                for (pos_name, image), tp in zip(tiff_positions, tiff_paths):
                    file_px = manual_px if manual_px > 0 else read_pixel_size_tiff(tp)
                    # Use the TIFF file stem as the file identifier so the `file`
                    # column in results.csv shows the actual filename, not the folder.
                    work_items.append((tp.stem, pos_name, image, file_px))
                if tiff_positions and manual_px == 0:
                    detected = work_items[0][3] if work_items else 0.0
                    if detected > 0:
                        self.progress.emit(
                            0,
                            1,
                            f"  Pixel size from TIFF metadata: {detected:.4f} µm/px",
                        )

            total = len(work_items)
            if total == 0:
                self.progress.emit(0, 1, "No images found — nothing to process.")
                self.finished.emit(pd.DataFrame(), [])
                return

            for pos_idx, (file_stem, pos_name, image, file_px) in enumerate(work_items):
                self.progress.emit(pos_idx, total, f"  {file_stem} | {pos_name}")
                safe_pos = _safe_filename(pos_name)

                # ── Save MIP TIFF (ND2 mode only — TIFFs are already MIPs) ───
                if source_type == "nd2":
                    mip_path = mip_dir / f"{file_stem}_{safe_pos}_MIP.tif"
                    try:
                        save_mip_tiff(image, mip_path)
                        self.progress.emit(
                            pos_idx,
                            total,
                            f"    MIP saved → mips/{mip_path.name}",
                        )
                    except Exception as exc:
                        self.progress.emit(
                            pos_idx,
                            total,
                            f"    WARNING: could not save MIP: {exc}",
                        )

                # ── Segment parasites ─────────────────────────────────────────
                # Convert µm² area limits to pixel² using per-file pixel size.
                # If pixel size is unknown, disable the area filter and warn.
                if file_px > 0:
                    min_area_px = min_area_um2 / (file_px**2)
                    max_area_px = max_area_um2 / (file_px**2)
                    self.progress.emit(
                        pos_idx,
                        total,
                        f"    Area filter: {min_area_um2}–{max_area_um2} µm² "
                        f"= {min_area_px:.0f}–{max_area_px:.0f} px² "
                        f"(pixel size {file_px} µm/px)",
                    )
                else:
                    min_area_px = 0.0
                    max_area_px = 1e9
                    self.progress.emit(
                        pos_idx,
                        total,
                        "    WARNING: pixel size unknown — area filter DISABLED. "
                        "Set pixel size in the UI to enforce µm² limits.",
                    )

                try:
                    labels, _, fstats = segment_pvs(
                        image=image,
                        channel_index=seg_ch,
                        use_composite=use_composite,
                        min_area_px=min_area_px,
                        max_area_px=max_area_px,
                        max_eccentricity=max_eccentricity,
                        min_solidity=min_solidity,
                        diameter=diameter,
                        flow_threshold=flow_threshold,
                        cellprob_threshold=cellprob_threshold,
                        threshold_method=p.get("threshold_method", "none"),
                        threshold_channel=p.get("threshold_channel", seg_ch),
                        threshold_value=p.get("threshold_value", 0.0),
                        threshold_percentile=p.get("threshold_percentile", 50.0),
                        watershed_split=p.get("watershed_split", False),
                        watershed_min_distance=p.get("watershed_min_distance", 10),
                    )
                    if fstats.get("threshold_skipped"):
                        self.progress.emit(
                            pos_idx,
                            total,
                            f"    WARNING: {fstats['threshold_method']} threshold "
                            f"(cutoff={fstats['threshold_cutoff']:.1f}) would remove "
                            f"{100 - fstats['pct_kept']:.1f}% of pixels — skipped.",
                        )
                    elif fstats["threshold_method"] != "none":
                        self.progress.emit(
                            pos_idx,
                            total,
                            f"    Threshold ({fstats['threshold_method']}, "
                            f"cutoff={fstats['threshold_cutoff']:.1f}): "
                            f"{fstats['pct_kept']:.1f}% pixels kept.",
                        )
                    self.progress.emit(
                        pos_idx,
                        total,
                        f"    Cellpose: {fstats['total_raw']} raw → "
                        f"{fstats['kept']} kept "
                        f"(area:{fstats['rejected_area']} "
                        f"ecc:{fstats['rejected_eccentricity']} "
                        f"sol:{fstats['rejected_solidity']} rejected)",
                    )
                except Exception as exc:
                    self.progress.emit(
                        pos_idx, total, f"    Segmentation failed: {exc}"
                    )
                    continue

                # ── Classifier filter ─────────────────────────────────────────
                if classifier is not None and labels.max() > 0:
                    features = extract_features(
                        labels=labels,
                        image=image,
                        seg_channel=seg_ch,
                        ch_cptsa=ch_cptsa,
                        ch_mcherry=ch_mcherry,
                        ch_names=ch_names,
                    )
                    if len(features) > 0:
                        labels = apply_classifier_filter(labels, features, classifier)

                n_parasites = int(labels.max())
                self.progress.emit(
                    pos_idx,
                    total,
                    f"    {n_parasites} parasites detected — measuring…",
                )

                # ── Save mask TIFF ────────────────────────────────────────────
                mask_path = mask_dir / f"{file_stem}_{safe_pos}_mask.tif"
                try:
                    save_mask_tiff(labels, mask_path)
                    self.progress.emit(
                        pos_idx,
                        total,
                        f"    Mask saved → masks/{mask_path.name}",
                    )
                except Exception as exc:
                    self.progress.emit(
                        pos_idx,
                        total,
                        f"    WARNING: could not save mask: {exc}",
                    )

                if n_parasites == 0:
                    continue

                # ── Measure fluorescence ──────────────────────────────────────
                try:
                    meas = measure_pvs(
                        labels=labels,
                        image=image,
                        ch_cptsa=ch_cptsa,
                        ch_mcherry=ch_mcherry,
                        ch_names=ch_names,
                        pixel_size_um=file_px if file_px > 0 else None,
                    )
                except Exception as exc:
                    self.progress.emit(pos_idx, total, f"    Measurement failed: {exc}")
                    continue

                # ── Group by vacuole, select one parasite each ────────────────
                if p.get("group_vacuoles", True) and not meas.empty:
                    vac_map = group_by_vacuole(
                        labels, dilation_px=p.get("dilation_px", 5)
                    )
                    meas = select_one_per_vacuole(
                        meas, vac_map, method=p.get("vacuole_method", "largest")
                    )

                # ── Store for curation gallery ────────────────────────────────
                curation_list.append(
                    {
                        "display_name": f"{file_stem} | {pos_name}",
                        "file": file_stem,
                        "position_name": pos_name,
                        "image": image,
                        "labels": labels,
                        "measurements": meas.copy(),
                    }
                )

                # ── Prepend metadata columns ──────────────────────────────────
                meas.insert(0, "position_name", pos_name)
                meas.insert(0, "position_index", pos_idx)
                meas.insert(0, "replicate", replicate)
                meas.insert(0, "cell_line", cell_line)
                meas.insert(0, "treatment", treatment)
                meas.insert(0, "file", file_stem)
                all_rows.append(meas)

            # ── Combine all positions ─────────────────────────────────────────
            if all_rows:
                result = pd.concat(all_rows, ignore_index=False)
                result = result.reset_index().rename(
                    columns={"label": "parasite_label"}
                )
            else:
                result = pd.DataFrame()

            self.finished.emit(result, curation_list)

        except Exception:
            self.error.emit(traceback.format_exc())


# ── Batch widget ──────────────────────────────────────────────────────────────


class BatchWidget(QWidget):
    """
    napari dock widget for bulk batch processing of fluorescence images.

    Workflow:
      1. Choose source: ND2 Z-stack files or a folder of TIFF Max IPs
      2. Fill in treatment, cell line, and replicate number
      3. Configure channel indices and segmentation parameters
      4. Choose an output folder
      5. Click Run

    For each position the pipeline produces:
      - (ND2 mode) A MIP TIFF in out_folder/mips/
      - A segmentation mask TIFF in out_folder/masks/
      - Rows appended to out_folder/results.csv

    After running, select any position and click 'Open in curation gallery'
    to review and accept/reject individual parasite segments.
    """

    def __init__(self, napari_viewer=None, parent=None):
        super().__init__(parent)
        self._viewer = napari_viewer  # used to open curation gallery
        self._classifier = None
        self._thread: QThread | None = None
        self._worker: _BatchWorker | None = None
        self._curation_data: list[dict] = []  # per-position data after batch run

        # ── Manual review state ───────────────────────────────────────────────
        # _result_df holds the raw batch output (all detected parasites) until
        # the user reviews and explicitly saves the filtered version.
        self._result_df: pd.DataFrame | None = None

        # Maps (file_stem, position_name) → {parasite_label: 0/1}
        # Only positions the user has opened in the curation gallery appear here.
        # Positions NOT present default to "keep all" when saving.
        self._curation_decisions: dict[tuple, dict] = {}

        self._build_ui()
        self._try_load_classifier()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        # Wrap everything in a scroll area so the panel is usable when
        # the dock widget is too short to show all controls.
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        outer.addWidget(scroll)

        _content = QWidget()
        scroll.setWidget(_content)

        root = QVBoxLayout(_content)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(8)

        # ---- Source type toggle ---------------------------------------------
        src_box = QGroupBox("Image source")
        src_layout = QVBoxLayout(src_box)

        src_type_row = QHBoxLayout()
        self._src_type = QComboBox()
        self._src_type.addItems(["ND2 files (Z-stacks)", "TIFF folder (Max IPs)"])
        src_type_row.addWidget(QLabel("Source type:"))
        src_type_row.addWidget(self._src_type, stretch=1)
        src_layout.addLayout(src_type_row)

        # ND2 file list
        self._nd2_widget = QWidget()
        nd2_layout = QVBoxLayout(self._nd2_widget)
        nd2_layout.setContentsMargins(0, 0, 0, 0)
        self._file_list = QListWidget()
        self._file_list.setSelectionMode(QListWidget.ExtendedSelection)
        self._file_list.setMaximumHeight(100)
        nd2_layout.addWidget(self._file_list)
        btn_row = QHBoxLayout()
        for label, slot in [
            ("Add files…", self._add_files),
            ("Add folder…", self._add_folder),
            ("Remove selected", self._remove_selected),
            ("Clear", self._file_list.clear),
        ]:
            b = QPushButton(label)
            b.clicked.connect(slot)
            btn_row.addWidget(b)
        nd2_layout.addLayout(btn_row)
        src_layout.addWidget(self._nd2_widget)

        # TIFF folder picker
        self._tiff_widget = QWidget()
        tiff_layout = QHBoxLayout(self._tiff_widget)
        tiff_layout.setContentsMargins(0, 0, 0, 0)
        self._tiff_folder = QLineEdit()
        self._tiff_folder.setPlaceholderText("Folder containing MIP TIFFs…")
        tiff_browse = QPushButton("Browse…")
        tiff_browse.clicked.connect(self._browse_tiff_folder)
        tiff_layout.addWidget(self._tiff_folder, stretch=1)
        tiff_layout.addWidget(tiff_browse)
        src_layout.addWidget(self._tiff_widget)

        root.addWidget(src_box)

        # Show/hide based on source type
        self._tiff_widget.setVisible(False)
        self._src_type.currentIndexChanged.connect(self._on_src_type_changed)

        # ---- Experiment metadata --------------------------------------------
        meta_box = QGroupBox("Experiment metadata")
        meta_form = QFormLayout(meta_box)
        meta_form.setContentsMargins(6, 6, 6, 6)

        self._treatment = QLineEdit()
        self._treatment.setPlaceholderText("e.g. DMSO, compound_X, untreated")
        meta_form.addRow("Treatment:", self._treatment)

        self._cell_line = QLineEdit()
        self._cell_line.setPlaceholderText("e.g. HFF, HeLa, RH")
        meta_form.addRow("Cell line:", self._cell_line)

        self._replicate = QSpinBox()
        self._replicate.setRange(1, 999)
        self._replicate.setValue(1)
        meta_form.addRow("Replicate #:", self._replicate)

        root.addWidget(meta_box)

        # ---- Channel configuration ------------------------------------------
        ch_box = QGroupBox("Channel configuration")
        ch_form = QFormLayout(ch_box)
        ch_form.setContentsMargins(6, 6, 6, 6)

        self._ch_cptsa = QSpinBox()
        self._ch_cptsa.setRange(0, 15)
        self._ch_cptsa.setValue(0)
        self._ch_cptsa.setToolTip("Channel index of cpTSapphire (NADH sensor)")
        ch_form.addRow("cpTSapphire channel:", self._ch_cptsa)

        self._ch_mcherry = QSpinBox()
        self._ch_mcherry.setRange(0, 15)
        self._ch_mcherry.setValue(1)
        self._ch_mcherry.setToolTip("Channel index of mCherry (reference)")
        ch_form.addRow("mCherry channel:", self._ch_mcherry)

        self._seg_ch = QSpinBox()
        self._seg_ch.setRange(0, 15)
        self._seg_ch.setValue(0)
        self._seg_ch.setToolTip(
            "Which channel Cellpose-SAM uses as input for segmentation"
        )
        self._use_composite = QCheckBox("Max composite of all channels")
        self._use_composite.setToolTip(
            "If checked, Cellpose-SAM receives the per-pixel channel maximum instead of "
            "a single channel."
        )
        seg_row = QHBoxLayout()
        seg_row.addWidget(self._seg_ch)
        seg_row.addWidget(self._use_composite)
        ch_form.addRow("Segment on channel:", seg_row)

        # Cellpose diameter
        self._diameter = QSpinBox()
        self._diameter.setRange(0, 2000)
        self._diameter.setValue(0)
        self._diameter.setSpecialValueText("auto")
        self._diameter.setToolTip(
            "Expected object diameter in pixels.  0 = auto-estimate.\n"
            "Increase this if cellpose finds individual parasites\n"
            "instead of whole vacuoles (try 30–80 for typical PVs)."
        )
        ch_form.addRow("Diameter (px, 0=auto):", self._diameter)

        # Cellpose thresholds
        self._flow_thresh = QDoubleSpinBox()
        self._flow_thresh.setRange(0.0, 3.0)
        self._flow_thresh.setSingleStep(0.1)
        self._flow_thresh.setDecimals(2)
        self._flow_thresh.setValue(0.4)
        self._flow_thresh.setToolTip(
            "Flow error threshold.  Higher = more permissive (more masks).  Default 0.4."
        )
        self._cellprob_thresh = QDoubleSpinBox()
        self._cellprob_thresh.setRange(-6.0, 6.0)
        self._cellprob_thresh.setSingleStep(0.5)
        self._cellprob_thresh.setDecimals(1)
        self._cellprob_thresh.setValue(0.0)
        self._cellprob_thresh.setToolTip(
            "Cell probability threshold.  Lower = more permissive.  Default 0.0."
        )
        thresh_row = QHBoxLayout()
        thresh_row.addWidget(QLabel("flow:"))
        thresh_row.addWidget(self._flow_thresh)
        thresh_row.addWidget(QLabel("prob:"))
        thresh_row.addWidget(self._cellprob_thresh)
        ch_form.addRow("Cellpose thresholds:", thresh_row)

        # Vacuole grouping
        self._group_vacuoles = QCheckBox("Group parasites into vacuoles")
        self._group_vacuoles.setChecked(True)
        self._group_vacuoles.setToolTip(
            "Merge nearby parasite masks to identify vacuoles, then select\n"
            "one representative parasite per vacuole for reporting."
        )
        self._dilation_px = QSpinBox()
        self._dilation_px.setRange(1, 100)
        self._dilation_px.setValue(5)
        self._dilation_px.setToolTip(
            "Dilation radius (px) used when grouping parasites into vacuoles."
        )
        self._vacuole_method = QComboBox()
        self._vacuole_method.addItems(["largest", "highest_ratio", "median_ratio"])
        self._vacuole_method.setToolTip(
            "Which parasite to use as the representative for each vacuole."
        )
        group_row = QHBoxLayout()
        group_row.addWidget(self._group_vacuoles)
        group_row.addWidget(QLabel("dilation:"))
        group_row.addWidget(self._dilation_px)
        ch_form.addRow("Vacuole grouping:", group_row)
        ch_form.addRow("Select parasite by:", self._vacuole_method)
        self._group_vacuoles.toggled.connect(self._dilation_px.setEnabled)
        self._group_vacuoles.toggled.connect(self._vacuole_method.setEnabled)

        # Watershed splitting
        self._watershed_split = QCheckBox("Split merged segments (watershed)")
        self._watershed_split.setChecked(False)
        self._watershed_split.setToolTip(
            "If Cellpose segments the whole vacuole as one object, enable this\n"
            "to split each mask into individual parasites using distance-transform\n"
            "watershed seeded by intensity peaks."
        )
        self._watershed_min_dist = QSpinBox()
        self._watershed_min_dist.setRange(2, 200)
        self._watershed_min_dist.setValue(10)
        self._watershed_min_dist.setToolTip(
            "Minimum pixel distance between neighbouring parasite centres.\n"
            "≈ parasite radius in pixels. At 0.105 µm/px (60×), 10 px ≈ 1 µm."
        )
        watershed_row = QHBoxLayout()
        watershed_row.addWidget(self._watershed_split)
        watershed_row.addWidget(QLabel("min sep:"))
        watershed_row.addWidget(self._watershed_min_dist)
        ch_form.addRow("Watershed split:", watershed_row)
        self._watershed_split.toggled.connect(self._watershed_min_dist.setEnabled)
        self._watershed_min_dist.setEnabled(False)

        # Pre-segmentation intensity threshold
        self._thresh_method = QComboBox()
        self._thresh_method.addItems(["none", "otsu", "percentile", "manual"])
        self._thresh_method.setToolTip(
            "Threshold applied before segmentation to mask out autofluorescent regions.\n"
            "  none        — no masking\n"
            "  otsu        — automatic Otsu threshold\n"
            "  percentile  — threshold at Nth percentile of non-zero pixels\n"
            "  manual      — explicit cutoff value"
        )
        self._thresh_channel = QSpinBox()
        self._thresh_channel.setRange(0, 15)
        self._thresh_channel.setValue(0)
        self._thresh_channel.setToolTip("Channel to compute the threshold on")
        self._thresh_value = QDoubleSpinBox()
        self._thresh_value.setRange(0.0, 1e9)
        self._thresh_value.setDecimals(1)
        self._thresh_value.setValue(0.0)
        self._thresh_value.setToolTip("Manual threshold value (raw image units)")
        self._thresh_percentile = QDoubleSpinBox()
        self._thresh_percentile.setRange(0.0, 100.0)
        self._thresh_percentile.setSingleStep(5.0)
        self._thresh_percentile.setDecimals(1)
        self._thresh_percentile.setValue(50.0)
        self._thresh_percentile.setToolTip(
            "Percentile of non-zero pixels to threshold at (0–100)"
        )
        thresh_method_row = QHBoxLayout()
        thresh_method_row.addWidget(self._thresh_method)
        thresh_method_row.addWidget(QLabel("ch:"))
        thresh_method_row.addWidget(self._thresh_channel)
        ch_form.addRow("Intensity threshold:", thresh_method_row)
        thresh_val_row = QHBoxLayout()
        thresh_val_row.addWidget(QLabel("value:"))
        thresh_val_row.addWidget(self._thresh_value)
        thresh_val_row.addWidget(QLabel("pct:"))
        thresh_val_row.addWidget(self._thresh_percentile)
        ch_form.addRow("", thresh_val_row)

        def _update_thresh_controls(method: str):
            self._thresh_value.setEnabled(method == "manual")
            self._thresh_percentile.setEnabled(method == "percentile")
            self._thresh_channel.setEnabled(method != "none")

        self._thresh_method.currentTextChanged.connect(_update_thresh_controls)
        _update_thresh_controls("none")

        self._min_area_um2 = QDoubleSpinBox()
        self._min_area_um2.setRange(0.0, 100_000.0)
        self._min_area_um2.setDecimals(1)
        self._min_area_um2.setValue(5.0)
        self._min_area_um2.setToolTip(
            "Minimum parasite area in µm². Smaller objects are discarded.\n"
            "Typical parasites at 60× with 0.105 µm/px: 5–25 µm².\n"
            "Converted to pixels using the pixel size below. Set to 0 to disable."
        )
        self._max_area_um2 = QDoubleSpinBox()
        self._max_area_um2.setRange(0.0, 100_000.0)
        self._max_area_um2.setDecimals(1)
        self._max_area_um2.setValue(25.0)
        self._max_area_um2.setToolTip(
            "Maximum parasite area in µm². Larger objects are discarded.\n"
            "Typical parasites at 60× with 0.105 µm/px: 5–25 µm².\n"
            "Converted to pixels using the pixel size below. Set to 100000 to disable."
        )
        area_row = QHBoxLayout()
        area_row.addWidget(QLabel("min:"))
        area_row.addWidget(self._min_area_um2)
        area_row.addWidget(QLabel("max:"))
        area_row.addWidget(self._max_area_um2)
        ch_form.addRow("Area filter (µm²):", area_row)

        self._max_eccentricity = QDoubleSpinBox()
        self._max_eccentricity.setRange(0.0, 1.0)
        self._max_eccentricity.setSingleStep(0.05)
        self._max_eccentricity.setDecimals(2)
        self._max_eccentricity.setValue(0.95)
        self._max_eccentricity.setToolTip(
            "Reject segments more elongated than this.\n"
            "0 = perfect circle, 1 = line. Parasites are typically < 0.85.\n"
            "Set to 1.0 to disable this filter."
        )
        self._min_solidity = QDoubleSpinBox()
        self._min_solidity.setRange(0.0, 1.0)
        self._min_solidity.setSingleStep(0.05)
        self._min_solidity.setDecimals(2)
        self._min_solidity.setValue(0.60)
        self._min_solidity.setToolTip(
            "Reject segments less solid than this (area / convex hull area).\n"
            "Parasites are compact (> 0.7); debris is often fragmented or irregular.\n"
            "Set to 0.0 to disable this filter."
        )
        shape_row = QHBoxLayout()
        shape_row.addWidget(QLabel("max ecc:"))
        shape_row.addWidget(self._max_eccentricity)
        shape_row.addWidget(QLabel("min sol:"))
        shape_row.addWidget(self._min_solidity)
        ch_form.addRow("Shape filter:", shape_row)

        self._pixel_size = QDoubleSpinBox()
        self._pixel_size.setRange(0, 100)
        self._pixel_size.setDecimals(4)
        self._pixel_size.setValue(0.0)
        self._pixel_size.setToolTip(
            "Physical pixel size in µm (used for area_um2 column). "
            "Leave at 0 to skip area calibration."
        )
        ch_form.addRow("Pixel size (µm):", self._pixel_size)

        root.addWidget(ch_box)

        # ---- Output folder --------------------------------------------------
        out_box = QGroupBox("Output")
        out_layout = QFormLayout(out_box)
        out_layout.setContentsMargins(6, 6, 6, 6)

        # Output folder — MIPs go in a mips/ subdirectory, CSV saved at root
        self._out_folder = QLineEdit()
        self._out_folder.setPlaceholderText("Select output folder…")
        browse_out = QPushButton("…")
        browse_out.setFixedWidth(28)
        browse_out.clicked.connect(self._browse_output_folder)
        out_row = QHBoxLayout()
        out_row.addWidget(self._out_folder, stretch=1)
        out_row.addWidget(browse_out)
        out_layout.addRow("Output folder:", out_row)

        # Annotations dir — used to load the trained PV classifier
        self._annot_dir = QLineEdit()
        self._annot_dir.setText(str(Path(__file__).parent.parent / "annotations"))
        browse_annot = QPushButton("…")
        browse_annot.setFixedWidth(28)
        browse_annot.clicked.connect(self._browse_annot_dir)
        annot_row = QHBoxLayout()
        annot_row.addWidget(self._annot_dir, stretch=1)
        annot_row.addWidget(browse_annot)
        out_layout.addRow("Annotations dir:", annot_row)

        root.addWidget(out_box)

        # ---- Classifier status ----------------------------------------------
        self._use_classifier = QCheckBox("Apply classifier filter")
        self._use_classifier.setChecked(False)
        self._use_classifier.setToolTip(
            "When checked, a trained RandomForest classifier automatically removes\n"
            "false-positive segments before measurements are computed.\n"
            "Uncheck until you have enough curated data for a reliable model."
        )
        root.addWidget(self._use_classifier)
        self._clf_label = QLabel("Classifier: not loaded")
        self._clf_label.setWordWrap(True)
        root.addWidget(self._clf_label)

        # ---- Run + progress -------------------------------------------------
        self._btn_run = QPushButton("▶ Run batch")
        self._btn_run.setStyleSheet("font-weight: bold;")
        self._btn_run.clicked.connect(self._run)
        root.addWidget(self._btn_run)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        root.addWidget(self._progress_bar)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(150)
        root.addWidget(self._log)

        # ---- Manual review + save -------------------------------------------
        review_box = QGroupBox("Manual review")
        review_layout = QVBoxLayout(review_box)
        review_layout.setContentsMargins(6, 6, 6, 6)

        review_info = QLabel(
            "After running, review each position in the curation gallery\n"
            "to accept/reject individual segments. Then save accepted results."
        )
        review_info.setWordWrap(True)
        review_layout.addWidget(review_info)

        self._review_status = QLabel("No batch run yet.")
        self._review_status.setWordWrap(True)
        review_layout.addWidget(self._review_status)

        self._curation_combo = QComboBox()
        self._curation_combo.setEnabled(False)
        review_layout.addWidget(self._curation_combo)

        self._btn_curate = QPushButton("Open selected position in curation gallery")
        self._btn_curate.setEnabled(False)
        self._btn_curate.clicked.connect(self._open_curation)
        review_layout.addWidget(self._btn_curate)

        self._btn_save_results = QPushButton("Save accepted results CSV")
        self._btn_save_results.setStyleSheet("font-weight: bold;")
        self._btn_save_results.setEnabled(False)
        self._btn_save_results.setToolTip(
            "Saves results.csv to the output folder.\n"
            "Parasites you explicitly rejected in the curation gallery are excluded.\n"
            "Positions not yet reviewed keep all detected parasites."
        )
        self._btn_save_results.clicked.connect(self._save_accepted_results)
        review_layout.addWidget(self._btn_save_results)

        root.addWidget(review_box)

        root.addStretch()

    # ── File list management ──────────────────────────────────────────────────

    def _add_files(self):
        """Open a file picker, add the selected ND2 files to the list."""
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select ND2 files", "", "ND2 files (*.nd2)"
        )
        self._add_paths(paths)

    def _add_folder(self):
        """Recursively find and add all ND2 files in a chosen folder."""
        folder = QFileDialog.getExistingDirectory(
            self, "Select folder containing ND2 files"
        )
        if folder:
            nd2_files = sorted(Path(folder).rglob("*.nd2"))
            self._add_paths([str(p) for p in nd2_files])
            self._log_msg(f"Added {len(nd2_files)} ND2 file(s) from {folder}")

    def _add_paths(self, paths: list[str]):
        """Add paths to the list widget, silently skipping duplicates."""
        existing = {
            self._file_list.item(i).text() for i in range(self._file_list.count())
        }
        for p in paths:
            if p not in existing:
                self._file_list.addItem(p)
                existing.add(p)

    def _remove_selected(self):
        """Remove highlighted items from the file list."""
        for item in self._file_list.selectedItems():
            self._file_list.takeItem(self._file_list.row(item))

    def _on_src_type_changed(self, index: int):
        """Show/hide ND2 list or TIFF folder picker based on source type."""
        is_nd2 = index == 0
        self._nd2_widget.setVisible(is_nd2)
        self._tiff_widget.setVisible(not is_nd2)

    def _browse_tiff_folder(self):
        """Let the user choose the folder of pre-made MIP TIFFs."""
        folder = QFileDialog.getExistingDirectory(
            self, "Select folder containing MIP TIFFs"
        )
        if folder:
            self._tiff_folder.setText(folder)

    # ── Directory pickers ─────────────────────────────────────────────────────

    def _browse_output_folder(self):
        """Let the user choose where MIPs and results.csv will be saved."""
        folder = QFileDialog.getExistingDirectory(self, "Select output folder")
        if folder:
            self._out_folder.setText(folder)

    def _browse_annot_dir(self):
        """Let the user point to a different annotations directory."""
        d = QFileDialog.getExistingDirectory(self, "Select annotations directory")
        if d:
            self._annot_dir.setText(d)
            self._try_load_classifier()

    # ── Classifier ────────────────────────────────────────────────────────────

    def _try_load_classifier(self):
        """
        Try to load the trained RandomForest classifier from the annotations dir.

        If found, Cellpose-SAM output will be pre-filtered before measurement,
        reducing false positives automatically.  The status label tells the user
        whether auto-filtering is active and how many training examples exist.
        """
        from ._learning import classifier_stats, load_classifier

        annot_dir = self._annot_dir.text()
        clf = load_classifier(annot_dir)
        self._classifier = clf

        csv_path = Path(annot_dir) / "curated_features.csv"
        stats = classifier_stats(csv_path)

        if clf is not None:
            self._clf_label.setText(
                f"Classifier loaded — {stats['total']} annotations "
                f"({stats['accepted']} accept / {stats['rejected']} reject). "
                f"False-positive filtering is active."
            )
        else:
            self._clf_label.setText(
                "No classifier found. All Cellpose-SAM segments will appear in results. "
                "Use the single-image widget to curate parasites and build training data."
            )

    # ── Batch run ─────────────────────────────────────────────────────────────

    def _run(self):
        """Validate inputs and launch the background batch worker."""
        out_folder_str = self._out_folder.text().strip()
        if not out_folder_str:
            self._log_msg("Choose an output folder before running.")
            return
        out_folder = Path(out_folder_str)

        is_nd2 = self._src_type.currentIndex() == 0
        if is_nd2:
            n_files = self._file_list.count()
            if n_files == 0:
                self._log_msg("No ND2 files — use 'Add files…' or 'Add folder…'.")
                return
            nd2_paths = [Path(self._file_list.item(i).text()) for i in range(n_files)]
            source_type = "nd2"
            tiff_folder_str = ""
        else:
            tiff_folder_str = self._tiff_folder.text().strip()
            if not tiff_folder_str or not Path(tiff_folder_str).is_dir():
                self._log_msg("Choose a valid folder containing TIFF files.")
                return
            nd2_paths = []
            source_type = "tiff_folder"

        # If pixel size is not set, ask the user.  The value is stored back into
        # the spinbox so the worker picks it up via self._pixel_size.value() below.
        if self._pixel_size.value() == 0:
            val, ok = QInputDialog.getDouble(
                self,
                "Pixel size not set",
                "Enter physical pixel size (µm/px).\n"
                "Required for the µm² area filter.\n"
                "Cancel or enter 0 to skip (area filter will be disabled for\n"
                "any file where metadata is also absent):",
                decimals=4,
                min=0.0,
                max=100.0,
                value=0.105,
            )
            if ok and val > 0:
                self._pixel_size.setValue(val)
                self._log_msg(f"Pixel size set to {val:.4f} µm/px.")

        params = {
            "source_type": source_type,
            "nd2_paths": nd2_paths,
            "tiff_folder": tiff_folder_str,
            "treatment": self._treatment.text().strip() or "unknown",
            "cell_line": self._cell_line.text().strip() or "unknown",
            "replicate": self._replicate.value(),
            "ch_cptsa": self._ch_cptsa.value(),
            "ch_mcherry": self._ch_mcherry.value(),
            "seg_ch": self._seg_ch.value(),
            "use_composite": self._use_composite.isChecked(),
            "min_area_um2": self._min_area_um2.value(),
            "max_area_um2": self._max_area_um2.value(),
            "max_eccentricity": self._max_eccentricity.value(),
            "min_solidity": self._min_solidity.value(),
            "pixel_size": self._pixel_size.value(),
            "classifier": (
                self._classifier if self._use_classifier.isChecked() else None
            ),
            "out_folder": out_folder,
            "diameter": self._diameter.value() if self._diameter.value() > 0 else None,
            "flow_threshold": self._flow_thresh.value(),
            "cellprob_threshold": self._cellprob_thresh.value(),
            "group_vacuoles": self._group_vacuoles.isChecked(),
            "dilation_px": self._dilation_px.value(),
            "vacuole_method": self._vacuole_method.currentText(),
            "watershed_split": self._watershed_split.isChecked(),
            "watershed_min_distance": self._watershed_min_dist.value(),
            "threshold_method": self._thresh_method.currentText(),
            "threshold_channel": self._thresh_channel.value(),
            "threshold_value": self._thresh_value.value(),
            "threshold_percentile": self._thresh_percentile.value(),
        }

        self._btn_run.setEnabled(False)
        self._btn_run.setText("Running…")
        self._progress_bar.setValue(0)
        source_desc = (
            f"{len(nd2_paths)} ND2 file(s)"
            if source_type == "nd2"
            else f"TIFF folder: {tiff_folder_str}"
        )
        self._log_msg(
            f"Batch started — {source_desc} | "
            f"treatment={params['treatment']} | "
            f"cell_line={params['cell_line']} | "
            f"replicate={params['replicate']}\n"
            f"Output folder: {out_folder}"
        )

        self._thread = QThread()
        self._worker = _BatchWorker(params)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.finished.connect(self._on_thread_done)

        # Store for use in _on_finished
        self._pending_out_folder = out_folder

        # Pre-load the Cellpose model in the main thread so the CUDA context is
        # established here before the worker thread starts.  The device name is
        # written to the log so slow runs can be diagnosed immediately.
        from ._segment import preload_model

        self._log_msg(preload_model())

        self._thread.start()

    # ── Worker callbacks ──────────────────────────────────────────────────────

    def _on_progress(self, current: int, total: int, msg: str):
        """Update the progress bar and log panel from the worker thread."""
        self._log_msg(msg)
        if total > 0:
            self._progress_bar.setValue(int(100 * current / total))

    def _on_finished(self, result: pd.DataFrame, curation_list: list):
        """
        Called in the main thread when the worker completes.

        Results are held in memory — the user must review positions in the
        curation gallery and click 'Save accepted results CSV' to write the
        final output.  This prevents false positives from polluting the CSV
        before manual QC.
        """
        self._progress_bar.setValue(100)

        if result is None or len(result) == 0:
            self._log_msg("Batch complete — no parasites detected.")
            self._review_status.setText("Batch complete — no parasites found.")
            return

        # Store raw results and reset any prior curation decisions
        self._result_df = result
        self._curation_decisions = {}

        n_parasites = len(result)
        n_positions = len(curation_list)

        # Populate curation gallery
        self._curation_data = curation_list
        self._curation_combo.clear()
        for item in curation_list:
            self._curation_combo.addItem(item["display_name"])

        if curation_list:
            self._curation_combo.setEnabled(True)
            self._btn_curate.setEnabled(True)
            self._btn_save_results.setEnabled(True)

        self._update_review_status()
        self._log_msg(
            f"Batch complete — {n_parasites} parasite(s) across {n_positions} position(s).\n"
            f"Review each position, then click 'Save accepted results CSV'."
        )

    def _on_error(self, tb: str):
        """Log an unhandled exception from the worker thread."""
        self._log_msg(f"BATCH ERROR:\n{tb}")

    def _on_thread_done(self):
        """Re-enable the Run button after the thread has fully stopped."""
        self._btn_run.setEnabled(True)
        self._btn_run.setText("▶ Run batch")

    def _update_review_status(self):
        """Refresh the '0/N reviewed' label and mark reviewed positions in the combo."""
        n_total = len(self._curation_data)
        n_reviewed = len(self._curation_decisions)
        self._review_status.setText(
            f"{n_reviewed}/{n_total} position(s) reviewed. "
            f"Unreviewed positions keep all detected parasites."
        )
        # Mark reviewed items in the combo with a checkmark
        for i, item in enumerate(self._curation_data):
            key = (item["file"], item["position_name"])
            base_name = item["display_name"]
            label = f"[✓] {base_name}" if key in self._curation_decisions else base_name
            self._curation_combo.setItemText(i, label)

    def _open_curation(self):
        """Open the selected position in the curation gallery."""
        idx = self._curation_combo.currentIndex()
        if idx < 0 or idx >= len(self._curation_data):
            return
        data = self._curation_data[idx]

        from ._curation import CurationWidget

        def _on_save(decisions: dict, vacuole_assignments: dict | None = None):
            from ._io import append_curated_annotations
            from ._learning import extract_features, train_classifier

            # Record decisions for this position so _save_accepted_results can
            # filter the result DataFrame.
            key = (data["file"], data["position_name"])
            self._curation_decisions[key] = decisions
            self._update_review_status()

            n_accepted = sum(1 for v in decisions.values() if v == 1)
            n_rejected = sum(1 for v in decisions.values() if v == 0)
            self._log_msg(
                f"Review saved — {data['display_name']}: "
                f"{n_accepted} accepted, {n_rejected} rejected."
            )

            annot_dir = self._annot_dir.text()
            feats = extract_features(
                labels=data["labels"],
                image=data["image"],
                seg_channel=self._seg_ch.value(),
                ch_cptsa=self._ch_cptsa.value(),
                ch_mcherry=self._ch_mcherry.value(),
                ch_names={
                    self._ch_cptsa.value(): "cptsa",
                    self._ch_mcherry.value(): "mcherry",
                },
            )
            append_curated_annotations(
                decisions=decisions,
                features=feats,
                image_stem=f"{data['file']}_{data['position_name']}",
                annotations_dir=annot_dir,
                vacuole_assignments=vacuole_assignments,
            )
            try:
                clf = train_classifier(Path(annot_dir) / "curated_features.csv")
                if clf is not None:
                    self._classifier = clf
                    self._log_msg("Classifier retrained from curation data.")
            except Exception as exc:
                self._log_msg(f"Classifier training skipped: {exc}")
            self._try_load_classifier()

        curation = CurationWidget(
            labels=data["labels"],
            image=data["image"],
            measurements=data["measurements"],
            ch_cptsa=self._ch_cptsa.value(),
            ch_mcherry=self._ch_mcherry.value(),
            on_save=_on_save,
        )
        curation.setWindowTitle(f"Curation — {data['display_name']}")
        curation.resize(340, 560)

        # Keep a strong reference so the widget is not garbage-collected when
        # this method returns (which would make the window vanish immediately).
        self._curation_win = curation

        # Dock into napari if viewer is available, otherwise show as window
        if self._viewer is not None:
            self._viewer.window.add_dock_widget(
                curation,
                name=f"Curation: {data['display_name']}",
                area="right",
            )
        else:
            curation.show()

    def _save_accepted_results(self):
        """
        Write results.csv containing only accepted parasites.

        For each position that was reviewed, parasites explicitly rejected
        (decision == 0) are excluded.  Positions not yet reviewed keep all
        detected parasites (default-accept policy — don't penalise partial QC).
        """
        if self._result_df is None or self._result_df.empty:
            self._log_msg("No results to save — run batch first.")
            return

        out_folder = self._pending_out_folder
        out_folder.mkdir(parents=True, exist_ok=True)

        df = self._result_df.copy()
        n_before = len(df)
        rows_to_drop = []

        for i, row in df.iterrows():
            key = (row["file"], row["position_name"])
            if key not in self._curation_decisions:
                continue  # not reviewed → keep
            decisions = self._curation_decisions[key]
            label = int(row["parasite_label"])
            if decisions.get(label) == 0:
                rows_to_drop.append(i)

        df = df.drop(index=rows_to_drop)
        n_after = len(df)
        n_rejected = n_before - n_after

        csv_path = out_folder / "results.csv"
        if csv_path.exists():
            df.to_csv(csv_path, mode="a", header=False, index=False)
            self._log_msg(
                f"Appended {n_after} row(s) to existing {csv_path} "
                f"({n_rejected} rejected excluded)."
            )
        else:
            df.to_csv(csv_path, index=False)
            self._log_msg(
                f"Saved {n_after} row(s) → {csv_path} ({n_rejected} rejected excluded)."
            )

    def _log_msg(self, msg: str):
        self._log.append(msg)


# ── napari entry point ────────────────────────────────────────────────────────


def make_batch_widget(napari_viewer=None):
    """
    Called by napari when the user opens 'Peredox: Batch Process ND2 Files'.
    """
    return BatchWidget(napari_viewer=napari_viewer)
