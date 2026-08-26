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
    QSizePolicy,
    QSpinBox,
    QTabWidget,
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
            analysis_mode: str = p.get("analysis_mode", "pv")
            treatment: str = p["treatment"]
            cell_line: str = p["cell_line"]
            replicate: int = p["replicate"]
            vac_seg_ch: int = p.get("vac_seg_ch", 1)  # Stage 1 — whole vacuole
            use_composite: bool = p["use_composite"]
            vac_min_area_um2: float = p.get("vac_min_area_um2", 20.0)
            vac_max_area_um2: float = p.get("vac_max_area_um2", 2000.0)
            max_eccentricity: float = p.get("max_eccentricity", 0.85)
            min_solidity: float = p.get("min_solidity", 0.70)
            pixel_size: float = p["pixel_size"]
            out_folder: Path = p["out_folder"]
            diameter = p.get("diameter")
            flow_threshold: float = p.get("flow_threshold", 0.4)
            cellprob_threshold: float = p.get("cellprob_threshold", 0.0)

            seg_backend: int = p.get("seg_backend", 0)  # 0=cpSAM, 1=StarDist
            annot_dir: str = p.get("annot_dir", "")

            # Pre-load StarDist vacuole model once before the position loop
            sd_vac_model = None
            if seg_backend == 1:
                from pathlib import Path as _Path

                from ._stardist import load_stardist_model

                sd_model_dir = _Path(annot_dir) / "stardist_model"
                sd_vac_model = load_stardist_model(sd_model_dir, mode="vacuoles")
                if sd_vac_model is None:
                    self.error.emit(
                        f"StarDist vacuole model not found in {sd_model_dir}. "
                        "Train the vacuole model first using the StarDist panel."
                    )
                    return
                self.progress.emit(0, 1, "StarDist vacuole model loaded.")

            # Create output subdirectories
            mip_dir = out_folder / "mips"
            mask_dir = out_folder / "masks"
            mip_dir.mkdir(parents=True, exist_ok=True)
            mask_dir.mkdir(parents=True, exist_ok=True)

            curation_list: list[dict] = []  # per-position data for curation gallery
            host_results: dict = {}  # (file_stem, pos_name) -> {"hosts": df, "parasites": df}

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

                if analysis_mode in ("host", "host_only"):
                    try:
                        self._process_host_position(
                            p,
                            image,
                            file_stem,
                            pos_name,
                            file_px,
                            pos_idx,
                            total,
                            mask_dir,
                            host_results,
                            curation_list,
                        )
                    except Exception as exc:
                        self.progress.emit(
                            pos_idx,
                            total,
                            f"    Host pipeline failed for this position: {exc}",
                        )
                    continue

                # ── Stage 1: detect whole vacuoles ───────────────────────────
                if file_px > 0:
                    vac_min_area_px = vac_min_area_um2 / (file_px**2)
                    vac_max_area_px = vac_max_area_um2 / (file_px**2)
                    self.progress.emit(
                        pos_idx,
                        total,
                        f"    Vacuole area filter: {vac_min_area_um2}–{vac_max_area_um2} µm² "
                        f"= {vac_min_area_px:.0f}–{vac_max_area_px:.0f} px²",
                    )
                else:
                    vac_min_area_px = 0.0
                    vac_max_area_px = 1e9

                try:
                    if seg_backend == 1:
                        from ._segment import filter_labels
                        from ._stardist import predict_stardist

                        raw_sd = predict_stardist(image, sd_vac_model, vac_seg_ch)
                        vac_labels, _, fstats = filter_labels(
                            raw_sd,
                            vac_min_area_px,
                            vac_max_area_px,
                            max_eccentricity,
                            min_solidity,
                        )
                        self.progress.emit(
                            pos_idx,
                            total,
                            f"    StarDist vacuoles: {fstats['total_raw']} raw → "
                            f"{fstats['kept']} kept "
                            f"(area:{fstats['rejected_area']} "
                            f"ecc:{fstats['rejected_eccentricity']} "
                            f"sol:{fstats['rejected_solidity']} rejected)",
                        )
                    else:
                        from ._segment import segment_pvs

                        vac_labels, _, fstats = segment_pvs(
                            image=image,
                            channel_index=vac_seg_ch,
                            use_composite=use_composite,
                            min_area_px=vac_min_area_px,
                            max_area_px=vac_max_area_px,
                            max_eccentricity=max_eccentricity,
                            min_solidity=min_solidity,
                            diameter=diameter,
                            flow_threshold=flow_threshold,
                            cellprob_threshold=cellprob_threshold,
                            threshold_method=p.get("threshold_method", "none"),
                            threshold_channel=p.get("threshold_channel", vac_seg_ch),
                            threshold_value=p.get("threshold_value", 0.0),
                            threshold_percentile=p.get("threshold_percentile", 50.0),
                            watershed_split=False,  # don't split vacuole masks
                        )
                        if fstats.get("threshold_skipped"):
                            self.progress.emit(
                                pos_idx,
                                total,
                                f"    WARNING: threshold would remove "
                                f"{100 - fstats['pct_kept']:.1f}% of pixels — skipped.",
                            )
                        self.progress.emit(
                            pos_idx,
                            total,
                            f"    cpSAM vacuoles: {fstats['total_raw']} raw → "
                            f"{fstats['kept']} kept "
                            f"(area:{fstats['rejected_area']} "
                            f"ecc:{fstats['rejected_eccentricity']} "
                            f"sol:{fstats['rejected_solidity']} rejected)",
                        )
                except Exception as exc:
                    self.progress.emit(
                        pos_idx, total, f"    Stage 1 segmentation failed: {exc}"
                    )
                    continue

                n_vacuoles = int(vac_labels.max())
                self.progress.emit(
                    pos_idx, total, f"    {n_vacuoles} vacuoles detected."
                )

                # ── Save vacuole mask TIFF ────────────────────────────────────
                mask_path = mask_dir / f"{file_stem}_{safe_pos}_vac_mask.tif"
                try:
                    save_mask_tiff(vac_labels, mask_path)
                    self.progress.emit(
                        pos_idx, total, f"    Vac mask → masks/{mask_path.name}"
                    )
                except Exception as exc:
                    self.progress.emit(
                        pos_idx, total, f"    WARNING: could not save vac mask: {exc}"
                    )

                # ── Store for curation gallery ────────────────────────────────
                # Parasite detection (Stage 2) runs interactively during curation.
                curation_list.append(
                    {
                        "display_name": f"{file_stem} | {pos_name}",
                        "file": file_stem,
                        "position_name": pos_name,
                        "image": image,
                        "vac_labels": vac_labels,
                        "file_px": file_px,
                        "treatment": treatment,
                        "cell_line": cell_line,
                        "replicate": replicate,
                        "pos_idx": pos_idx,
                    }
                )

            # Worker emits an empty DataFrame for now — results are built during
            # interactive curation and saved via _save_accepted_results.
            if analysis_mode in ("host", "host_only"):
                self.finished.emit(host_results, curation_list)
            else:
                self.finished.emit(pd.DataFrame(), curation_list)

        except Exception:
            self.error.emit(traceback.format_exc())

    def _process_host_position(
        self,
        p,
        image,
        file_stem,
        pos_name,
        file_px,
        pos_idx,
        total,
        mask_dir,
        host_results,
        curation_list,
    ):
        """Full auto host pipeline for one position: H1 → H2 → H3 (spec §5).

        In host-only mode (analysis_mode == "host_only") Stage H2 is skipped
        entirely: no parasite detection, no assignment, and the hosts table is
        exported without the parasite-assessment columns.
        """
        from ._host import (
            assign_to_hosts,
            drop_infection_columns,
            measure_hosts,
            segment_host_cells,
        )
        from ._measure import measure_pvs
        from ._segment import segment_parasites_in_vacuoles, segment_pvs

        host_only = p.get("analysis_mode") == "host_only"
        if p.get("seg_backend", 0) == 1 and not host_only:
            self.progress.emit(
                pos_idx,
                total,
                "    Note: StarDist backend is not supported in batch host "
                "mode — using cpSAM.",
            )

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
        # Host diameter comes from the Host-box spinbox, never from the
        # parasite/vacuole diameter (which is hidden in host-only mode).
        host_labels, _, hstats = segment_host_cells(
            image=image,
            channel_index=p.get("host_ch", ch_mcherry),
            clip_percentile=p.get("clip_percentile", 99.0),
            diameter=p.get("host_diameter"),
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
                labels=host_labels,
                image=image,
                seg_channel=p.get("host_ch", ch_mcherry),
                ch_cptsa=ch_cptsa,
                ch_mcherry=ch_mcherry,
                ch_names=ch_names,
            )
            host_labels = apply_classifier_filter(host_labels, feats, host_clf)
        n_hosts = len(np.unique(host_labels)) - 1
        self.progress.emit(
            pos_idx,
            total,
            f"    Hosts: {hstats['total_raw']} raw → {n_hosts} kept.",
        )
        if n_hosts == 0:
            self.progress.emit(pos_idx, total, "    No hosts — position skipped.")
            return

        # ── Stage H2: parasites inside hosts (auto) ──────────────────────────
        if host_only:
            # Host-only mode: parasites are never looked for.
            para_labels = np.zeros_like(host_labels)
            vac_map: dict[int, int] = {}
        else:
            masked = image * (host_labels > 0)[..., np.newaxis].astype(image.dtype)
            vac_labels, _, _ = segment_pvs(
                image=masked,
                channel_index=p.get("vac_seg_ch", 1),
                use_composite=p.get("use_composite", False),
                min_area_px=vac_min_px,
                max_area_px=vac_max_px,
                max_eccentricity=p.get("max_eccentricity", 0.85),
                min_solidity=p.get("min_solidity", 0.70),
                diameter=None,
                flow_threshold=p.get("flow_threshold", 0.4),
                cellprob_threshold=p.get("cellprob_threshold", 0.0),
                threshold_method=p.get("threshold_method", "none"),
                threshold_channel=p.get("threshold_channel", p.get("vac_seg_ch", 1)),
                threshold_value=p.get("threshold_value", 0.0),
                threshold_percentile=p.get("threshold_percentile", 50.0),
            )
            para_labels, vac_map = segment_parasites_in_vacuoles(
                image=masked,
                vac_labels=vac_labels,
                seg_channel=p.get("seg_ch", 0),
                model=None,
                min_area_px=p.get("min_area_um2", 5.0) / (file_px**2)
                if file_px > 0
                else 0.0,
                max_area_px=p.get("max_area_um2", 200.0) / (file_px**2)
                if file_px > 0
                else 1e9,
                max_eccentricity=p.get("max_eccentricity", 0.85),
                min_solidity=p.get("min_solidity", 0.70),
            )

        # ── Stage H3: assign + measure ───────────────────────────────────────
        if host_only:
            para_to_host: dict[int, int] = {}
            vac_to_host: dict[int, int] = {}
        else:
            para_to_host, vac_to_host, dropped = assign_to_hosts(
                para_labels, host_labels, vac_map
            )
            if dropped:
                self.progress.emit(
                    pos_idx,
                    total,
                    f"    {len(dropped)} parasite(s) without host majority "
                    f"dropped: {sorted(dropped)}",
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
        if host_only:
            # Parasites were never assessed — the infection columns would read
            # as "verified uninfected", so they are dropped instead.
            hosts_df = drop_infection_columns(hosts_df)
            para_df = pd.DataFrame()
        else:
            para_df = measure_pvs(
                labels=para_labels,
                image=image,
                ch_cptsa=ch_cptsa,
                ch_mcherry=ch_mcherry,
                ch_names=ch_names,
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

        if host_only:
            self.progress.emit(
                pos_idx,
                total,
                f"    {len(hosts_df)} hosts measured (host-only).",
            )
        else:
            n_inf = int(hosts_df["infected"].sum()) if not hosts_df.empty else 0
            self.progress.emit(
                pos_idx,
                total,
                f"    {len(hosts_df)} hosts ({n_inf} infected), "
                f"{len(para_df)} parasites.",
            )

        # ── Persist masks; stash results + curation entry ────────────────────
        try:
            save_mask_tiff(
                host_labels, mask_dir / f"{file_stem}_{safe_pos}_host_mask.tif"
            )
            if not host_only:
                save_mask_tiff(
                    para_labels,
                    mask_dir / f"{file_stem}_{safe_pos}_host_para_mask.tif",
                )
        except Exception as exc:
            self.progress.emit(pos_idx, total, f"    WARNING: mask save failed: {exc}")

        key = (file_stem, pos_name)
        host_results[key] = {"hosts": hosts_df, "parasites": para_df}
        curation_list.append(
            {
                "mode": "host",
                "host_only": host_only,
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
                # Channels as used by THIS run — the review save must not read
                # live spinboxes, which the user may have retuned for a later
                # run before saving a pending review.
                "host_ch": p.get("host_ch", ch_mcherry),
                "ch_cptsa": ch_cptsa,
                "ch_mcherry": ch_mcherry,
            }
        )


# ── Batch widget ──────────────────────────────────────────────────────────────

# Combo-index → internal mode string for the Channels-tab "Analysis mode"
# selector.  Order must match the addItems() call in _build_channels_tab.
_ANALYSIS_MODES = ("pv", "host", "host_only")


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

        # Host mode: {(file, position_name) -> {"hosts": df, "parasites": df}}
        # populated by _on_finished (host branch) and refined per-position by
        # _open_host_position_curation as the user reviews host masks.
        self._host_results: dict[tuple, dict] = {}

        # Maps (file_stem, position_name) → {parasite_label: 0/1}
        # Only positions the user has opened in the curation gallery appear here.
        # Positions NOT present default to "keep all" when saving.
        self._curation_decisions: dict[tuple, dict] = {}

        self._build_ui()
        self._try_load_classifier()
        self._update_stardist_status()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        self.setMinimumWidth(400)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(4)

        tabs = QTabWidget()
        tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        tabs.addTab(self._build_source_tab(), "Source")
        tabs.addTab(self._build_channels_tab(), "Channels")
        tabs.addTab(self._build_output_tab(), "Output")
        tabs.addTab(self._build_log_tab(), "Log")
        outer.addWidget(tabs)

    # ── Source tab ────────────────────────────────────────────────────────────

    def _build_source_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

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
        self._file_list.setMinimumHeight(80)
        nd2_layout.addWidget(self._file_list)
        btn_row = QHBoxLayout()
        for label, slot in [
            ("Add files…", self._add_files),
            ("Add folder…", self._add_folder),
            ("Remove", self._remove_selected),
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

        self._tiff_widget.setVisible(False)
        self._src_type.currentIndexChanged.connect(self._on_src_type_changed)
        layout.addWidget(src_box)

        # Experiment metadata
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

        layout.addWidget(meta_box)
        layout.addStretch()
        return w

    # ── Channels tab ─────────────────────────────────────────────────────────

    def _build_channels_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        # Analysis mode selector — governs which setting groups below are shown
        mode_box = QGroupBox("Analysis mode")
        mode_layout = QVBoxLayout(mode_box)
        self._analysis_mode = QComboBox()
        self._analysis_mode.addItems(
            ["Parasites / PVs", "Host cells + parasites", "Host cells only"]
        )
        self._analysis_mode.setToolTip(
            "Parasites / PVs — the original two-stage vacuole→parasite pipeline.\n"
            "Host cells + parasites — segment Peredox-expressing hosts, then\n"
            "parasites inside them (hosts.csv + host_parasites.csv).\n"
            "Host cells only — segment and measure hosts, skip parasite\n"
            "detection entirely (hosts.csv without infection columns)."
        )
        mode_layout.addWidget(self._analysis_mode)
        layout.addWidget(mode_box)

        # Ratio channels — needed by every analysis mode
        ratio_box = QGroupBox("Ratio channels")
        ratio_form = QFormLayout(ratio_box)
        ratio_form.setContentsMargins(6, 6, 6, 6)

        self._ch_cptsa = QSpinBox()
        self._ch_cptsa.setRange(0, 15)
        self._ch_cptsa.setValue(0)
        ratio_form.addRow("cpTSapphire ch:", self._ch_cptsa)

        self._ch_mcherry = QSpinBox()
        self._ch_mcherry.setRange(0, 15)
        self._ch_mcherry.setValue(1)
        ratio_form.addRow("mCherry ch:", self._ch_mcherry)

        layout.addWidget(ratio_box)

        # Parasite / vacuole segmentation — hidden in host-only mode
        ch_box = QGroupBox("Parasite / vacuole segmentation")
        ch_form = QFormLayout(ch_box)
        ch_form.setContentsMargins(6, 6, 6, 6)

        self._seg_backend = QComboBox()
        self._seg_backend.addItems(["cpSAM (Cellpose)", "StarDist (fine-tuned)"])
        ch_form.addRow("Model:", self._seg_backend)

        self._vac_seg_ch = QSpinBox()
        self._vac_seg_ch.setRange(0, 15)
        self._vac_seg_ch.setValue(1)
        self._vac_seg_ch.setToolTip("Channel used for Stage 1 whole-vacuole detection")
        ch_form.addRow("Vacuole ch:", self._vac_seg_ch)

        self._seg_ch = QSpinBox()
        self._seg_ch.setRange(0, 15)
        self._seg_ch.setValue(0)
        self._use_composite = QCheckBox("Max composite")
        seg_row = QHBoxLayout()
        seg_row.addWidget(self._seg_ch)
        seg_row.addWidget(self._use_composite)
        ch_form.addRow("Parasite ch:", seg_row)

        self._diameter = QSpinBox()
        self._diameter.setRange(0, 2000)
        self._diameter.setValue(0)
        self._diameter.setSpecialValueText("auto")
        ch_form.addRow("Diameter (px):", self._diameter)

        self._flow_thresh = QDoubleSpinBox()
        self._flow_thresh.setRange(0.0, 3.0)
        self._flow_thresh.setSingleStep(0.1)
        self._flow_thresh.setDecimals(2)
        self._flow_thresh.setValue(0.4)
        self._cellprob_thresh = QDoubleSpinBox()
        self._cellprob_thresh.setRange(-6.0, 6.0)
        self._cellprob_thresh.setSingleStep(0.5)
        self._cellprob_thresh.setDecimals(1)
        self._cellprob_thresh.setValue(0.0)
        thresh_row = QHBoxLayout()
        thresh_row.addWidget(QLabel("flow:"))
        thresh_row.addWidget(self._flow_thresh)
        thresh_row.addWidget(QLabel("prob:"))
        thresh_row.addWidget(self._cellprob_thresh)
        ch_form.addRow("cpSAM thresholds:", thresh_row)

        # Vacuole grouping
        self._group_vacuoles = QCheckBox("Group into vacuoles")
        self._group_vacuoles.setChecked(True)
        self._dilation_px = QSpinBox()
        self._dilation_px.setRange(1, 100)
        self._dilation_px.setValue(5)
        self._vacuole_method = QComboBox()
        self._vacuole_method.addItems(
            ["largest", "highest_ratio", "median_ratio", "mean_ratio"]
        )
        group_row = QHBoxLayout()
        group_row.addWidget(self._group_vacuoles)
        group_row.addWidget(QLabel("dil:"))
        group_row.addWidget(self._dilation_px)
        ch_form.addRow("Vacuole grouping:", group_row)
        ch_form.addRow("Select by:", self._vacuole_method)
        self._group_vacuoles.toggled.connect(self._dilation_px.setEnabled)
        self._group_vacuoles.toggled.connect(self._vacuole_method.setEnabled)

        # Watershed
        self._watershed_split = QCheckBox("Watershed split")
        self._watershed_split.setChecked(False)
        self._watershed_min_dist = QSpinBox()
        self._watershed_min_dist.setRange(2, 200)
        self._watershed_min_dist.setValue(10)
        ws_row = QHBoxLayout()
        ws_row.addWidget(self._watershed_split)
        ws_row.addWidget(QLabel("min sep:"))
        ws_row.addWidget(self._watershed_min_dist)
        ch_form.addRow("", ws_row)
        self._watershed_split.toggled.connect(self._watershed_min_dist.setEnabled)
        self._watershed_min_dist.setEnabled(False)

        # Intensity threshold
        self._thresh_method = QComboBox()
        self._thresh_method.addItems(["none", "otsu", "percentile", "manual"])
        self._thresh_channel = QSpinBox()
        self._thresh_channel.setRange(0, 15)
        self._thresh_value = QDoubleSpinBox()
        self._thresh_value.setRange(0.0, 1e9)
        self._thresh_value.setDecimals(1)
        self._thresh_percentile = QDoubleSpinBox()
        self._thresh_percentile.setRange(0.0, 100.0)
        self._thresh_percentile.setSingleStep(5.0)
        self._thresh_percentile.setDecimals(1)
        self._thresh_percentile.setValue(50.0)
        thr1 = QHBoxLayout()
        thr1.addWidget(self._thresh_method)
        thr1.addWidget(QLabel("ch:"))
        thr1.addWidget(self._thresh_channel)
        ch_form.addRow("Int. threshold:", thr1)
        thr2 = QHBoxLayout()
        thr2.addWidget(QLabel("val:"))
        thr2.addWidget(self._thresh_value)
        thr2.addWidget(QLabel("pct:"))
        thr2.addWidget(self._thresh_percentile)
        ch_form.addRow("", thr2)

        def _upd_thresh(method: str) -> None:
            self._thresh_value.setEnabled(method == "manual")
            self._thresh_percentile.setEnabled(method == "percentile")
            self._thresh_channel.setEnabled(method != "none")

        self._thresh_method.currentTextChanged.connect(_upd_thresh)
        _upd_thresh("none")

        self._cpsam_only_widgets = [
            self._use_composite,
            self._diameter,
            self._flow_thresh,
            self._cellprob_thresh,
            self._watershed_split,
            self._watershed_min_dist,
            self._thresh_method,
            self._thresh_channel,
            self._thresh_value,
            self._thresh_percentile,
        ]

        def _on_backend_changed(_idx: int) -> None:
            is_cpsam = self._seg_backend.currentIndex() == 0
            for ww in self._cpsam_only_widgets:
                ww.setEnabled(is_cpsam)

        self._seg_backend.currentIndexChanged.connect(_on_backend_changed)
        layout.addWidget(ch_box)

        # Morphology filters
        filt_box = QGroupBox("Morphology filters")
        filt_form = QFormLayout(filt_box)
        filt_form.setContentsMargins(6, 6, 6, 6)

        self._vac_min_area_um2 = QDoubleSpinBox()
        self._vac_min_area_um2.setRange(0.0, 1_000_000.0)
        self._vac_min_area_um2.setDecimals(1)
        self._vac_min_area_um2.setValue(20.0)
        self._vac_max_area_um2 = QDoubleSpinBox()
        self._vac_max_area_um2.setRange(0.0, 1_000_000.0)
        self._vac_max_area_um2.setDecimals(1)
        self._vac_max_area_um2.setValue(2000.0)
        vac_area_row = QHBoxLayout()
        vac_area_row.addWidget(QLabel("min:"))
        vac_area_row.addWidget(self._vac_min_area_um2)
        vac_area_row.addWidget(QLabel("max:"))
        vac_area_row.addWidget(self._vac_max_area_um2)
        filt_form.addRow("Vacuole area (µm²):", vac_area_row)

        self._min_area_um2 = QDoubleSpinBox()
        self._min_area_um2.setRange(0.0, 100_000.0)
        self._min_area_um2.setDecimals(1)
        self._min_area_um2.setValue(5.0)
        self._max_area_um2 = QDoubleSpinBox()
        self._max_area_um2.setRange(0.0, 100_000.0)
        self._max_area_um2.setDecimals(1)
        self._max_area_um2.setValue(25.0)
        area_row = QHBoxLayout()
        area_row.addWidget(QLabel("min:"))
        area_row.addWidget(self._min_area_um2)
        area_row.addWidget(QLabel("max:"))
        area_row.addWidget(self._max_area_um2)
        filt_form.addRow("Parasite area (µm²):", area_row)

        self._max_eccentricity = QDoubleSpinBox()
        self._max_eccentricity.setRange(0.0, 1.0)
        self._max_eccentricity.setSingleStep(0.05)
        self._max_eccentricity.setDecimals(2)
        self._max_eccentricity.setValue(0.95)
        self._min_solidity = QDoubleSpinBox()
        self._min_solidity.setRange(0.0, 1.0)
        self._min_solidity.setSingleStep(0.05)
        self._min_solidity.setDecimals(2)
        self._min_solidity.setValue(0.60)
        shape_row = QHBoxLayout()
        shape_row.addWidget(QLabel("max ecc:"))
        shape_row.addWidget(self._max_eccentricity)
        shape_row.addWidget(QLabel("min sol:"))
        shape_row.addWidget(self._min_solidity)
        filt_form.addRow("Shape:", shape_row)

        layout.addWidget(filt_box)

        # Host mode settings (enabled only when Analysis mode == "Host cells")
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

        self._host_diameter = QSpinBox()
        self._host_diameter.setRange(0, 2000)
        self._host_diameter.setValue(300)
        self._host_diameter.setSpecialValueText("auto")
        self._host_diameter.setToolTip(
            "Expected host-cell diameter in px (default 300 ≈ 32 µm at 60×).\n"
            "0 = auto — NOT recommended: on dim Peredox images auto-estimation\n"
            "shatters cells into speckle. Independent of the parasite/vacuole\n"
            "diameter above, which is hidden in host-only mode."
        )
        host_form.addRow("Host diameter (px):", self._host_diameter)

        self._host_min_area_um2 = QDoubleSpinBox()
        self._host_min_area_um2.setRange(0.0, 1e6)
        self._host_min_area_um2.setDecimals(0)
        # 350 µm² sits above a U2OS nucleus (~150-250 µm²) so nucleus-scale
        # junk is rejected while whole cells (~600-900 µm²) pass.
        self._host_min_area_um2.setValue(350.0)
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

        layout.addWidget(host_box)

        # Per-mode group visibility (the mode combo lives at the top of this tab)
        self._pv_seg_box = ch_box
        self._filt_box = filt_box
        self._host_box = host_box
        self._analysis_mode.currentIndexChanged.connect(
            lambda _i: self._update_mode_visibility()
        )
        self._update_mode_visibility()

        layout.addStretch()
        return w

    def _current_analysis_mode(self) -> str:
        """Return 'pv', 'host', or 'host_only' from the Channels-tab selector."""
        combo = getattr(self, "_analysis_mode", None)
        if combo is None:
            return "pv"
        idx = combo.currentIndex()
        if 0 <= idx < len(_ANALYSIS_MODES):
            return _ANALYSIS_MODES[idx]
        return "pv"

    def _update_mode_visibility(self) -> None:
        """Show only the Channels-tab groups relevant to the selected mode."""
        mode = self._current_analysis_mode()
        self._pv_seg_box.setVisible(mode != "host_only")
        self._filt_box.setVisible(mode != "host_only")
        self._host_box.setVisible(mode != "pv")
        # Refresh the classifier label to describe the mode's classifier.
        # Guarded: the first call fires during _build_channels_tab, before the
        # Output tab (which owns _clf_label) has been built.
        if getattr(self, "_clf_label", None) is not None:
            self._try_load_classifier()

    # ── Output tab ────────────────────────────────────────────────────────────

    def _build_output_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        # Pixel size + dirs
        out_box = QGroupBox("Output settings")
        out_form = QFormLayout(out_box)
        out_form.setContentsMargins(6, 6, 6, 6)

        self._pixel_size = QDoubleSpinBox()
        self._pixel_size.setRange(0, 100)
        self._pixel_size.setDecimals(4)
        self._pixel_size.setValue(0.0)
        self._pixel_size.setToolTip(
            "Physical pixel size in µm. Leave at 0 to auto-detect from file metadata."
        )
        out_form.addRow("Pixel size (µm):", self._pixel_size)

        self._out_folder = QLineEdit()
        self._out_folder.setPlaceholderText("Select output folder…")
        browse_out = QPushButton("…")
        browse_out.setFixedWidth(26)
        browse_out.clicked.connect(self._browse_output_folder)
        out_row = QHBoxLayout()
        out_row.addWidget(self._out_folder, stretch=1)
        out_row.addWidget(browse_out)
        out_form.addRow("Output folder:", out_row)

        self._annot_dir = QLineEdit()
        self._annot_dir.setText(str(Path(__file__).parent.parent / "annotations"))
        browse_annot = QPushButton("…")
        browse_annot.setFixedWidth(26)
        browse_annot.clicked.connect(self._browse_annot_dir)
        annot_row = QHBoxLayout()
        annot_row.addWidget(self._annot_dir, stretch=1)
        annot_row.addWidget(browse_annot)
        out_form.addRow("Annotations dir:", annot_row)

        layout.addWidget(out_box)

        # Classifier
        clf_box = QGroupBox("Classifier")
        clf_layout = QVBoxLayout(clf_box)
        self._use_classifier = QCheckBox("Apply classifier filter")
        self._use_classifier.setChecked(False)
        clf_layout.addWidget(self._use_classifier)
        self._clf_label = QLabel("Classifier: not loaded")
        self._clf_label.setWordWrap(True)
        clf_layout.addWidget(self._clf_label)
        layout.addWidget(clf_box)

        # Run button + progress
        run_box = QGroupBox("Run")
        run_layout = QVBoxLayout(run_box)
        self._btn_run = QPushButton("▶ Run batch")
        self._btn_run.setStyleSheet("font-weight: bold;")
        self._btn_run.clicked.connect(self._run)
        run_layout.addWidget(self._btn_run)
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        run_layout.addWidget(self._progress_bar)
        layout.addWidget(run_box)

        # Review + save
        review_box = QGroupBox("Review & save")
        review_layout = QVBoxLayout(review_box)
        self._review_status = QLabel("No batch run yet.")
        self._review_status.setWordWrap(True)
        review_layout.addWidget(self._review_status)

        self._curation_combo = QComboBox()
        self._curation_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._curation_combo.setEnabled(False)
        review_layout.addWidget(self._curation_combo)

        self._btn_curate = QPushButton("Open in curation gallery")
        self._btn_curate.setEnabled(False)
        self._btn_curate.clicked.connect(self._open_curation)
        review_layout.addWidget(self._btn_curate)

        self._btn_save_results = QPushButton("💾 Save accepted results CSV")
        self._btn_save_results.setStyleSheet("font-weight: bold;")
        self._btn_save_results.setEnabled(False)
        self._btn_save_results.clicked.connect(self._save_accepted_results)
        review_layout.addWidget(self._btn_save_results)

        layout.addWidget(review_box)
        layout.addStretch()
        return w

    # ── Log tab ───────────────────────────────────────────────────────────────

    def _build_log_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(6, 6, 6, 6)

        # StarDist training
        sd_box = QGroupBox("StarDist training")
        sd_layout = QVBoxLayout(sd_box)
        self._sd_status = QLabel("Training data: — pairs")
        self._sd_status.setWordWrap(True)
        sd_layout.addWidget(self._sd_status)

        sd_cfg = QFormLayout()
        sd_cfg.setContentsMargins(0, 0, 0, 0)
        self._sd_seg_ch = QSpinBox()
        self._sd_seg_ch.setRange(0, 7)
        sd_cfg.addRow("Seg channel:", self._sd_seg_ch)
        self._sd_epochs = QSpinBox()
        self._sd_epochs.setRange(10, 1000)
        self._sd_epochs.setValue(100)
        sd_cfg.addRow("Epochs:", self._sd_epochs)
        sd_layout.addLayout(sd_cfg)

        sd_btn_row = QHBoxLayout()
        self._btn_sd_train = QPushButton("Train StarDist model")
        self._btn_sd_train.setStyleSheet("font-weight: bold;")
        self._btn_sd_train.clicked.connect(self._train_stardist)
        self._btn_sd_refresh = QPushButton("↺ Refresh")
        self._btn_sd_refresh.clicked.connect(self._update_stardist_status)
        sd_btn_row.addWidget(self._btn_sd_train)
        sd_btn_row.addWidget(self._btn_sd_refresh)
        sd_layout.addLayout(sd_btn_row)
        layout.addWidget(sd_box)

        # Log
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        layout.addWidget(self._log)

        return w

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
            self._update_stardist_status()

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

        # The status label describes whichever classifier the selected
        # analysis mode will actually apply: PV classifier in PV mode, the
        # separate host classifier (curated_host_features.*) in host modes.
        if self._current_analysis_mode() != "pv":
            host_clf = load_classifier(
                annot_dir, filename="curated_host_features.joblib"
            )
            stats = classifier_stats(Path(annot_dir) / "curated_host_features.csv")
            if host_clf is not None:
                self._clf_label.setText(
                    f"Host classifier loaded — {stats['total']} annotations "
                    f"({stats['accepted']} accept / {stats['rejected']} reject). "
                    f"Stage H1 false-positive filtering is active."
                )
            else:
                self._clf_label.setText(
                    "No host classifier yet. Review host cells here or in the "
                    "single-image Host tab to build its training data "
                    "(curated_host_features.csv, separate from the PV classifier)."
                )
            return

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

    def _update_stardist_status(self):
        """Refresh the StarDist training-data pair count label."""
        from ._stardist import count_training_pairs

        annot_dir = self._annot_dir.text()
        if not annot_dir:
            self._sd_status.setText("Training data: set annotations dir first.")
            return
        training_dir = Path(annot_dir) / "training_data"
        n_vac = count_training_pairs(training_dir, mode="vacuoles")
        n_para = count_training_pairs(training_dir, mode="parasites")
        model_dir = Path(annot_dir) / "stardist_model"
        vac_ready = (model_dir / "vacuoles" / "peredox_vacuoles").exists()
        para_ready = (model_dir / "parasites" / "peredox_parasites").exists()
        status = (
            f"Vacuoles: {n_vac} pairs {'✓' if vac_ready else '(untrained)'}  |  "
            f"Parasites: {n_para} pairs {'✓' if para_ready else '(untrained)'}"
        )
        self._sd_status.setText(status)

    def _train_stardist(self):
        """Launch StarDist fine-tuning in a background thread."""
        from ._stardist import count_training_pairs

        annot_dir = self._annot_dir.text()
        if not annot_dir:
            self._log_msg("Set the annotations directory before training.")
            return

        training_dir = Path(annot_dir) / "training_data"
        n_pairs = count_training_pairs(training_dir, mode="parasites")
        if n_pairs < 5:
            self._log_msg(
                f"Only {n_pairs} parasite training pair(s) — curate more images first "
                f"(need at least 5; 20+ recommended)."
            )
            return

        self._btn_sd_train.setEnabled(False)
        self._log_msg(
            f"Starting StarDist training on {n_pairs} pairs "
            f"({self._sd_epochs.value()} epochs)…"
        )

        from qtpy.QtCore import QObject, QThread, Signal

        class _StarDistWorker(QObject):
            progress = Signal(str)
            finished = Signal(str)  # model path or ""
            error = Signal(str)

            def __init__(self, params):
                super().__init__()
                self.params = params

            def run(self):
                try:
                    from napari_peredox._stardist import train_stardist

                    p = self.params
                    out = train_stardist(
                        training_dir=p["training_dir"],
                        model_dir=p["model_dir"],
                        seg_channel=p["seg_channel"],
                        n_epochs=p["n_epochs"],
                        progress_cb=self.progress.emit,
                    )
                    self.finished.emit(str(out))
                except Exception:
                    import traceback

                    self.error.emit(traceback.format_exc())

        params = {
            "training_dir": training_dir,
            "model_dir": Path(annot_dir) / "stardist_model",
            "seg_channel": self._sd_seg_ch.value(),
            "n_epochs": self._sd_epochs.value(),
        }
        self._sd_thread = QThread()
        self._sd_worker = _StarDistWorker(params)
        self._sd_worker.moveToThread(self._sd_thread)
        self._sd_thread.started.connect(self._sd_worker.run)
        self._sd_worker.progress.connect(self._log_msg)
        self._sd_worker.finished.connect(self._on_stardist_done)
        self._sd_worker.error.connect(self._on_stardist_error)
        self._sd_worker.finished.connect(self._sd_thread.quit)
        self._sd_worker.error.connect(self._sd_thread.quit)
        self._sd_thread.finished.connect(lambda: self._btn_sd_train.setEnabled(True))
        self._sd_thread.start()

    def _on_stardist_done(self, model_path: str):
        self._log_msg(f"StarDist training complete — model saved to {model_path}")
        self._update_stardist_status()

    def _on_stardist_error(self, tb: str):
        self._log_msg(f"StarDist training failed:\n{tb}")

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

        analysis_mode = self._current_analysis_mode()
        # In host-only mode the parasite/vacuole settings group is hidden —
        # its widgets must not silently influence host segmentation, so the
        # shared cpSAM flow/cellprob fall back to their defaults there.
        hidden_pv_settings = analysis_mode == "host_only"

        params = {
            "source_type": source_type,
            "nd2_paths": nd2_paths,
            "tiff_folder": tiff_folder_str,
            "treatment": self._treatment.text().strip() or "unknown",
            "cell_line": self._cell_line.text().strip() or "unknown",
            "replicate": self._replicate.value(),
            "ch_cptsa": self._ch_cptsa.value(),
            "ch_mcherry": self._ch_mcherry.value(),
            "vac_seg_ch": self._vac_seg_ch.value(),
            "seg_ch": self._seg_ch.value(),
            "use_composite": self._use_composite.isChecked(),
            "vac_min_area_um2": self._vac_min_area_um2.value(),
            "vac_max_area_um2": self._vac_max_area_um2.value(),
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
            "flow_threshold": 0.4 if hidden_pv_settings else self._flow_thresh.value(),
            "cellprob_threshold": (
                0.0 if hidden_pv_settings else self._cellprob_thresh.value()
            ),
            "group_vacuoles": self._group_vacuoles.isChecked(),
            "dilation_px": self._dilation_px.value(),
            "vacuole_method": self._vacuole_method.currentText(),
            "watershed_split": self._watershed_split.isChecked(),
            "watershed_min_distance": self._watershed_min_dist.value(),
            "threshold_method": self._thresh_method.currentText(),
            "threshold_channel": self._thresh_channel.value(),
            "threshold_value": self._thresh_value.value(),
            "threshold_percentile": self._thresh_percentile.value(),
            "seg_backend": self._seg_backend.currentIndex(),  # 0=cpSAM, 1=StarDist
            "annot_dir": self._annot_dir.text(),
            "analysis_mode": analysis_mode,
            "host_diameter": (
                float(self._host_diameter.value())
                if self._host_diameter.value() > 0
                else None
            ),
            "host_ch": getattr(self, "_host_ch", None).value()
            if getattr(self, "_host_ch", None)
            else 1,
            "clip_percentile": getattr(self, "_host_clip_pct", None).value()
            if getattr(self, "_host_clip_pct", None)
            else 99.0,
            "host_min_area_um2": getattr(self, "_host_min_area_um2", None).value()
            if getattr(self, "_host_min_area_um2", None)
            else 350.0,
            "host_max_area_um2": getattr(self, "_host_max_area_um2", None).value()
            if getattr(self, "_host_max_area_um2", None)
            else 10000.0,
            "host_dilation_px": getattr(self, "_host_dilation_px", None).value()
            if getattr(self, "_host_dilation_px", None)
            else 3,
            "host_classifier": self._load_host_classifier_if_requested(),
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

        # Host modes always run Cellpose for Stage H1, regardless of the
        # (possibly hidden) StarDist backend selection — preload it so the
        # CUDA context is established in the main thread.
        if self._seg_backend.currentIndex() == 1 and analysis_mode == "pv":
            self._log_msg("Using fine-tuned StarDist model for segmentation.")
        else:
            # Pre-load the Cellpose model in the main thread so the CUDA context is
            # established here before the worker thread starts.
            from ._segment import preload_model

            self._log_msg(preload_model())

        self._thread.start()

    # ── Worker callbacks ──────────────────────────────────────────────────────

    def _on_progress(self, current: int, total: int, msg: str):
        """Update the progress bar and log panel from the worker thread."""
        self._log_msg(msg)
        if total > 0:
            self._progress_bar.setValue(int(100 * current / total))

    def _on_finished(self, result: pd.DataFrame | dict, curation_list: list):
        """
        Called in the main thread when the worker completes.

        Results are held in memory — the user must review positions in the
        curation gallery and click 'Save accepted results CSV' to write the
        final output.  This prevents false positives from polluting the CSV
        before manual QC.

        `result` is a `pd.DataFrame` in PV mode (all detected parasites) or a
        `dict` keyed by `(file, position_name) -> {"hosts", "parasites"}` in
        host mode (see `_process_host_position`).
        """
        if isinstance(result, dict):
            # Host mode: `result` is {(file, pos) -> {"hosts", "parasites"}}.
            # An empty dict means every position was skipped (no hosts found
            # or the per-position pipeline failed) — curation_list is then
            # empty too, so there's nothing to review or save.
            if not result:
                self._result_df = None
                self._host_results = {}
                self._curation_data = []
                self._curation_combo.clear()
                self._progress_bar.setValue(100)
                self._log_msg(
                    "Host batch complete — no host cells detected in any position."
                )
                return
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

        self._progress_bar.setValue(100)

        # Store raw results and reset any prior curation decisions
        self._result_df = result
        self._curation_decisions = {}
        # Clear stale host results so a later Save doesn't re-write the
        # previous host batch's CSVs instead of this PV batch's results.
        self._host_results = {}

        n_positions = len(curation_list)

        if n_positions == 0:
            self._log_msg("Batch complete — no vacuoles detected in any position.")
            self._review_status.setText("Batch complete — no vacuoles found.")
            return

        # Populate curation gallery
        self._curation_data = curation_list
        self._curation_combo.clear()
        for item in curation_list:
            self._curation_combo.addItem(item["display_name"])

        self._curation_combo.setEnabled(True)
        self._btn_curate.setEnabled(True)
        self._btn_save_results.setEnabled(True)

        self._update_review_status()
        self._log_msg(
            f"Stage 1 complete — {n_positions} position(s) with vacuoles detected.\n"
            f"Open each position in the curation gallery to review vacuoles, "
            f"then parasite detection runs interactively during curation."
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
        if self._curation_data and self._curation_data[0].get("mode") == "host":
            kept = "host cells"
        else:
            kept = "parasites"
        self._review_status.setText(
            f"{n_reviewed}/{n_total} position(s) reviewed. "
            f"Unreviewed positions keep all detected {kept}."
        )
        # Mark reviewed items in the combo with a checkmark
        for i, item in enumerate(self._curation_data):
            key = (item["file"], item["position_name"])
            base_name = item["display_name"]
            label = f"[✓] {base_name}" if key in self._curation_decisions else base_name
            self._curation_combo.setItemText(i, label)

    def _open_curation(self):
        """
        Open two-pass curation for the selected position.

        Pass 1 — VacuoleCurationWidget: user reviews whole-vacuole outlines,
                  accepts/rejects/redraws each detected vacuole.
        Pass 2 — CurationWidget: after saving vacuole decisions, opens the
                  per-vacuole parasite gallery for individual parasite review.
        """
        idx = self._curation_combo.currentIndex()
        if idx < 0 or idx >= len(self._curation_data):
            return
        item = self._curation_data[idx]
        if item.get("mode") == "host":
            self._open_host_position_curation(item)
            return

        idx = self._curation_combo.currentIndex()
        if idx < 0 or idx >= len(self._curation_data):
            return
        data = self._curation_data[idx]

        ch_names = {
            self._ch_cptsa.value(): "cptsa",
            self._ch_mcherry.value(): "mcherry",
        }
        px = self._pixel_size.value()
        annot_dir = self._annot_dir.text()
        stem = f"{data['file']}_{data['position_name']}"
        display_name = data["display_name"]

        # Load image+labels into the napari viewer for context
        vac_layer_name = display_name.replace(" ", "_") + "_vacuoles"
        if self._viewer is not None:
            img_hwc = data["image"]
            img_chw = np.moveaxis(img_hwc, -1, 0) if img_hwc.ndim == 3 else img_hwc
            img_layer_name = display_name.replace(" ", "_") + "_image"
            if img_layer_name in self._viewer.layers:
                self._viewer.layers[img_layer_name].data = img_chw
            else:
                self._viewer.add_image(img_chw, name=img_layer_name, channel_axis=0)
            if vac_layer_name in self._viewer.layers:
                self._viewer.layers[vac_layer_name].data = data["vac_labels"]
            else:
                self._viewer.add_labels(data["vac_labels"], name=vac_layer_name)
            self._viewer.reset_view()

        # ── Pass 2 callback — opens after vacuole curation is saved ──────────
        def _open_parasite_curation(vac_decisions: dict, curated_vac_labels):
            from qtpy.QtWidgets import QMessageBox

            from ._curation import CurationWidget
            from ._segment import segment_parasites_in_vacuoles

            # Diagnostic: log the full decision map and what's in the label array
            ids_in_labels = [int(v) for v in np.unique(curated_vac_labels) if v != 0]
            self._log_msg(
                f"  [diag] vac_decisions: { {k: v for k, v in vac_decisions.items()} }"
            )
            self._log_msg(f"  [diag] IDs in curated_vac_labels: {ids_in_labels}")

            # Keep only explicitly accepted vacuoles (dec == 1).
            # Skipped (dec == -1) and rejected (dec == 0) are both zeroed out.
            accepted_vac = curated_vac_labels.copy()
            for vid in np.unique(curated_vac_labels):
                if vid == 0:
                    continue
                if vac_decisions.get(int(vid), -1) != 1:
                    accepted_vac[accepted_vac == vid] = 0

            ids_accepted = [int(v) for v in np.unique(accepted_vac) if v != 0]
            self._log_msg(f"  [diag] IDs after accept filter: {ids_accepted}")

            n_accepted = int(len([d for d in vac_decisions.values() if d == 1]))
            if n_accepted == 0:
                self._log_msg(
                    f"{display_name}: all vacuoles rejected — nothing to segment."
                )
                return

            self._log_msg(
                f"{display_name}: {n_accepted} vacuoles accepted — running parasite detection…"
            )

            # Run Stage 2 parasite segmentation on accepted vacuoles
            try:
                seg_backend = self._seg_backend.currentIndex()
                sd_model = None
                if seg_backend == 0:
                    # Ensure cpSAM model is loaded in the main thread
                    from ._segment import preload_model

                    self._log_msg(preload_model())
                else:
                    from ._stardist import load_stardist_model

                    sd_model = load_stardist_model(
                        Path(annot_dir) / "stardist_model", mode="parasites"
                    )
                file_px = data.get("file_px", px)
                min_area_px = (
                    self._min_area_um2.value() / (file_px**2) if file_px > 0 else 0.0
                )
                max_area_px = (
                    self._max_area_um2.value() / (file_px**2) if file_px > 0 else 1e9
                )
                para_labels, vac_map = segment_parasites_in_vacuoles(
                    image=data["image"],
                    vac_labels=accepted_vac,
                    seg_channel=self._seg_ch.value(),
                    model=sd_model,
                    min_area_px=min_area_px,
                    max_area_px=max_area_px,
                    max_eccentricity=self._max_eccentricity.value(),
                    min_solidity=self._min_solidity.value(),
                    progress_cb=self._log_msg,
                )
            except Exception as exc:
                import traceback as _tb

                msg = f"Parasite segmentation failed: {exc}"
                self._log_msg(msg)
                self._log_msg(_tb.format_exc())
                QMessageBox.critical(self, "Segmentation error", msg)
                return

            # Log per-vacuole parasite counts so missing vacuoles are visible
            vac_para_counts: dict[int, int] = {}
            for pid, vid in vac_map.items():
                vac_para_counts[vid] = vac_para_counts.get(vid, 0) + 1
            for vid in ids_accepted:
                count = vac_para_counts.get(vid, 0)
                self._log_msg(f"  [diag] vac {vid}: {count} parasite(s) found")

            # Measure all parasites — pass the full per-parasite table to
            # CurationWidget so every vacuole appears in the gallery.
            # select_one_per_vacuole runs later in _on_para_save, after curation.
            from ._measure import measure_pvs

            try:
                meas = measure_pvs(
                    labels=para_labels,
                    image=data["image"],
                    ch_cptsa=self._ch_cptsa.value(),
                    ch_mcherry=self._ch_mcherry.value(),
                    ch_names=ch_names,
                    pixel_size_um=px if px > 0 else None,
                )
            except Exception as exc:
                self._log_msg(f"Measurement failed: {exc}")
                meas = None

            # Update napari layer to show parasite labels
            para_layer_name = display_name.replace(" ", "_") + "_parasites"
            if self._viewer is not None:
                if para_layer_name in self._viewer.layers:
                    self._viewer.layers[para_layer_name].data = para_labels
                else:
                    self._viewer.add_labels(para_labels, name=para_layer_name)

            def _on_para_save(decisions: dict, vacuole_assignments: dict | None = None):
                from ._io import append_curated_annotations, save_training_pair
                from ._learning import extract_features, train_classifier
                from ._measure import measure_pvs

                key = (data["file"], data["position_name"])
                self._curation_decisions[key] = decisions
                data["final_labels"] = para_labels
                data["final_vac_map"] = vac_map
                self._update_review_status()

                n_acc = sum(1 for v in decisions.values() if v == 1)
                n_rej = sum(1 for v in decisions.values() if v == 0)
                self._log_msg(
                    f"Parasite review saved — {display_name}: "
                    f"{n_acc} accepted, {n_rej} rejected."
                )

                # Measure the whole-PV region (Stage 1 vacuole mask) — one row
                # per vacuole.  This gives the correct Peredox ratio because the
                # mCherry signal fills the entire PV lumen, not just parasite blobs.
                try:
                    file_px = data.get("file_px", px)

                    # Count accepted parasites per vacuole and collect per-parasite
                    # ratios so we can compute mean/median per vacuole.
                    accepted_para = {lbl for lbl, dec in decisions.items() if dec == 1}
                    para_count_per_vac: dict[int, int] = {}
                    para_ratios_per_vac: dict[int, list[float]] = {}
                    if vac_map:
                        # Measure individual accepted parasites for ratio aggregation
                        meas_para = measure_pvs(
                            labels=para_labels,
                            image=data["image"],
                            ch_cptsa=self._ch_cptsa.value(),
                            ch_mcherry=self._ch_mcherry.value(),
                            ch_names=ch_names,
                            pixel_size_um=file_px if file_px > 0 else None,
                        )
                        for plbl, vid in vac_map.items():
                            if plbl not in accepted_para:
                                continue
                            para_count_per_vac[vid] = para_count_per_vac.get(vid, 0) + 1
                            if not meas_para.empty and plbl in meas_para.index:
                                r = meas_para.loc[plbl, "ratio_cptsa_mcherry"]
                                if not np.isnan(r):
                                    para_ratios_per_vac.setdefault(vid, []).append(
                                        float(r)
                                    )

                    # Measure on vacuole mask — one row per PV, whole-lumen ratio
                    # (accepted_vac is the filtered Stage 1 label array in scope)
                    meas_vac = measure_pvs(
                        labels=accepted_vac,
                        image=data["image"],
                        ch_cptsa=self._ch_cptsa.value(),
                        ch_mcherry=self._ch_mcherry.value(),
                        ch_names=ch_names,
                        pixel_size_um=file_px if file_px > 0 else None,
                    )

                    if not meas_vac.empty:
                        # Parasite count and per-parasite ratio aggregates
                        meas_vac["parasites_per_vacuole"] = meas_vac.index.map(
                            lambda vid: para_count_per_vac.get(vid, 0)
                        )
                        meas_vac["mean_parasite_ratio"] = meas_vac.index.map(
                            lambda vid: (
                                float(np.mean(para_ratios_per_vac[vid]))
                                if vid in para_ratios_per_vac
                                else np.nan
                            )
                        )
                        meas_vac["median_parasite_ratio"] = meas_vac.index.map(
                            lambda vid: (
                                float(np.median(para_ratios_per_vac[vid]))
                                if vid in para_ratios_per_vac
                                else np.nan
                            )
                        )
                        meas_vac.insert(0, "position_name", data["position_name"])
                        meas_vac.insert(0, "replicate", data.get("replicate", 1))
                        meas_vac.insert(0, "cell_line", data.get("cell_line", ""))
                        meas_vac.insert(0, "treatment", data.get("treatment", ""))
                        meas_vac.insert(0, "file", data["file"])
                        meas_vac = meas_vac.reset_index().rename(
                            columns={"label": "vacuole_id"}
                        )
                        if self._result_df is None or self._result_df.empty:
                            self._result_df = meas_vac
                        else:
                            self._result_df = pd.concat(
                                [self._result_df, meas_vac], ignore_index=True
                            )
                        self._log_msg(
                            f"  Measurements accumulated — {len(self._result_df)} total rows."
                        )
                except Exception as exc:
                    self._log_msg(f"  Measurement failed: {exc}")

                feats = extract_features(
                    labels=para_labels,
                    image=data["image"],
                    seg_channel=self._seg_ch.value(),
                    ch_cptsa=self._ch_cptsa.value(),
                    ch_mcherry=self._ch_mcherry.value(),
                    ch_names=ch_names,
                )
                append_curated_annotations(
                    decisions=decisions,
                    features=feats,
                    image_stem=stem,
                    annotations_dir=annot_dir,
                    vacuole_assignments=vacuole_assignments,
                )
                try:
                    from ._stardist import count_training_pairs

                    training_dir = Path(annot_dir) / "training_data"
                    save_training_pair(
                        image=data["image"],
                        labels=para_labels,
                        decisions=decisions,
                        stem=stem,
                        training_dir=training_dir,
                        vacuole_map=vac_map or None,
                    )
                    nv = count_training_pairs(training_dir, mode="vacuoles")
                    np_ = count_training_pairs(training_dir, mode="parasites")
                    self._log_msg(
                        f"Training pairs saved — vacuoles: {nv}, parasites: {np_}"
                    )
                    self._update_stardist_status()
                except Exception as exc2:
                    self._log_msg(f"Training pair save skipped: {exc2}")
                try:
                    clf = train_classifier(Path(annot_dir) / "curated_features.csv")
                    if clf is not None:
                        self._classifier = clf
                except Exception:
                    pass
                self._try_load_classifier()

            para_win = CurationWidget(
                labels=para_labels,
                image=data["image"],
                measurements=meas,
                ch_cptsa=self._ch_cptsa.value(),
                ch_mcherry=self._ch_mcherry.value(),
                ch_names=ch_names,
                pixel_size_um=px if px > 0 else None,
                vacuole_assignments=vac_map or None,
                accepted_vac_ids=ids_accepted,
                vac_labels=accepted_vac,
                on_save=_on_para_save,
                viewer=self._viewer,
                labels_layer_name=para_layer_name,
            )
            para_win.setWindowTitle(f"Parasites — {display_name}")
            para_win.resize(360, 580)
            self._curation_win2 = para_win  # keep strong ref

            if self._viewer is not None:
                self._viewer.window.add_dock_widget(
                    para_win,
                    name=f"Parasites: {display_name}",
                    area="right",
                )
            else:
                para_win.show()

        # ── Pass 1 — vacuole curation ─────────────────────────────────────────
        from ._curation import VacuoleCurationWidget

        def _on_vac_save(vac_decisions: dict, curated_vac_labels):
            n_vac_acc = sum(1 for d in vac_decisions.values() if d == 1)
            self._log_msg(
                f"Vacuole review saved — {display_name}: {n_vac_acc} vacuoles accepted."
            )
            # Save vacuole training pair
            try:
                from ._io import save_training_pair
                from ._stardist import count_training_pairs

                training_dir = Path(annot_dir) / "training_data"
                save_training_pair(
                    image=data["image"],
                    labels=curated_vac_labels,
                    decisions=vac_decisions,
                    stem=stem + "_vac",
                    training_dir=training_dir,
                    vacuole_map={
                        int(v): int(v) for v in np.unique(curated_vac_labels) if v != 0
                    },
                )
                nv = count_training_pairs(training_dir, mode="vacuoles")
                self._log_msg(f"Vacuole training pairs: {nv}")
                self._update_stardist_status()
            except Exception as exc:
                self._log_msg(f"Vacuole training pair skipped: {exc}")
            # Defer opening parasite curation to the next event-loop tick so
            # the vacuole widget's save signal has fully unwound before we
            # open a new dock widget.
            from qtpy.QtCore import QTimer

            QTimer.singleShot(
                0,
                lambda vd=vac_decisions, cvl=curated_vac_labels: (
                    _open_parasite_curation(vd, cvl)
                ),
            )

        vac_win = VacuoleCurationWidget(
            vac_labels=data["vac_labels"],
            image=data["image"],
            ch_cptsa=self._ch_cptsa.value(),
            ch_mcherry=self._ch_mcherry.value(),
            on_save=_on_vac_save,
            viewer=self._viewer,
            labels_layer_name=vac_layer_name,
        )
        vac_win.setWindowTitle(f"Vacuoles — {display_name}")
        vac_win.resize(360, 520)
        self._curation_win = vac_win  # keep strong ref

        if self._viewer is not None:
            self._viewer.window.add_dock_widget(
                vac_win,
                name=f"Vacuoles: {display_name}",
                area="right",
            )
        else:
            vac_win.show()

    def _open_host_position_curation(self, item: dict):
        """Accept/reject/redraw host masks for one batch position."""
        from ._curation import VacuoleCurationWidget

        def _on_save(decisions: dict, curated_hosts: np.ndarray):
            from ._host import assign_to_hosts, drop_infection_columns, measure_hosts

            host_only = bool(item.get("host_only"))
            hosts = curated_hosts.copy()
            for hid, dec in decisions.items():
                if dec == 0:
                    hosts[hosts == hid] = 0
            item["host_labels"] = hosts

            # Recompute assignment + measurement against the curated hosts.
            # Parasites in rejected hosts lose their majority and are dropped.
            if host_only:
                para_to_host: dict[int, int] = {}
                vac_to_host: dict[int, int] = {}
            else:
                para_to_host, vac_to_host, _dropped = assign_to_hosts(
                    item["para_labels"], hosts, item.get("vac_map") or None
                )
            # Channels as captured at run time — never the live spinboxes,
            # which the user may have retuned for a later run before saving
            # this (asynchronous) review.
            ch_cptsa = item.get("ch_cptsa", self._ch_cptsa.value())
            ch_mcherry = item.get("ch_mcherry", self._ch_mcherry.value())
            ch_names = {0: "ch0", 1: "ch1"}
            ch_names[ch_cptsa] = "cptsa"
            ch_names[ch_mcherry] = "mcherry"
            file_px = item["file_px"]
            hosts_df = measure_hosts(
                host_labels=hosts,
                para_labels=item["para_labels"],
                image=item["image"],
                para_to_host=para_to_host,
                vac_to_host=vac_to_host,
                dilation_px=self._host_dilation_px.value(),
                ch_cptsa=ch_cptsa,
                ch_mcherry=ch_mcherry,
                ch_names=ch_names,
                pixel_size_um=file_px if file_px > 0 else None,
            )
            if host_only:
                hosts_df = drop_infection_columns(hosts_df)
                para_df = pd.DataFrame()
            else:
                from ._measure import measure_pvs

                para_df = measure_pvs(
                    labels=item["para_labels"],
                    image=item["image"],
                    ch_cptsa=ch_cptsa,
                    ch_mcherry=ch_mcherry,
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

            # Grow the host classifier's training set from this review — the
            # same curated_host_features.* files the single-image Host tab
            # feeds, kept fully separate from the PV classifier.  Features
            # come from the pre-rejection array so rejected hosts contribute
            # negative examples.
            try:
                from ._io import append_curated_annotations
                from ._learning import extract_features, train_classifier

                feats = extract_features(
                    labels=curated_hosts,
                    image=item["image"],
                    seg_channel=item.get("host_ch", self._host_ch.value()),
                    ch_cptsa=ch_cptsa,
                    ch_mcherry=ch_mcherry,
                    ch_names=ch_names,
                )
                host_stem = (
                    f"{item['file']}_{_safe_filename(item['position_name'])}_host"
                )
                csv_path = append_curated_annotations(
                    decisions=decisions,
                    features=feats,
                    image_stem=host_stem,
                    annotations_dir=self._annot_dir.text(),
                    csv_name="curated_host_features.csv",
                )
                n_dec = sum(1 for v in decisions.values() if v in (0, 1))
                self._log_msg(f"Saved {n_dec} host annotations → {csv_path}")
                if train_classifier(csv_path) is not None:
                    self._log_msg("Host classifier retrained.")
                else:
                    self._log_msg("Not enough host data to train the classifier yet.")
                self._try_load_classifier()
            except Exception as exc:
                self._log_msg(f"Host annotation save error: {exc}")

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
        self._host_curation_win.setWindowTitle(f"Host Review — {item['display_name']}")
        self._host_curation_win.resize(360, 560)
        self._host_curation_win.show()

    def _save_accepted_results(self):
        """Write results.csv from measurements accumulated during curation."""
        if getattr(self, "_host_results", None):
            out_folder = self._pending_out_folder
            hosts_list = [
                v["hosts"] for v in self._host_results.values() if not v["hosts"].empty
            ]
            if not hosts_list:
                self._log_msg(
                    "No host rows to save — review at least one position with "
                    "accepted hosts."
                )
                return
            out_folder.mkdir(parents=True, exist_ok=True)
            hosts_all = pd.concat(hosts_list)
            paras = [
                v["parasites"]
                for v in self._host_results.values()
                if v["parasites"] is not None and not v["parasites"].empty
            ]
            hosts_path = out_folder / "hosts.csv"
            if hosts_path.exists():
                self._log_msg(f"Overwriting existing {hosts_path.name}.")
            hosts_all.to_csv(hosts_path)
            self._log_msg(f"Saved {len(hosts_all)} host row(s) → {hosts_path}")
            if paras:
                paras_all = pd.concat(paras)
                paras_path = out_folder / "host_parasites.csv"
                if paras_path.exists():
                    self._log_msg(f"Overwriting existing {paras_path.name}.")
                paras_all.to_csv(paras_path)
                self._log_msg(f"Saved {len(paras_all)} parasite row(s) → {paras_path}")
            return

        if self._result_df is None or self._result_df.empty:
            self._log_msg(
                "No results to save — complete curation for at least one position first."
            )
            return

        out_folder = self._pending_out_folder
        out_folder.mkdir(parents=True, exist_ok=True)

        df = self._result_df.copy()
        csv_path = out_folder / "results.csv"
        if csv_path.exists():
            df.to_csv(csv_path, mode="a", header=False, index=False)
            self._log_msg(f"Appended {len(df)} row(s) to existing {csv_path}.")
        else:
            df.to_csv(csv_path, index=False)
            self._log_msg(f"Saved {len(df)} row(s) → {csv_path}.")

    def _load_host_classifier_if_requested(self):
        """Host classifier for batch host mode; None when unavailable/not requested."""
        if self._current_analysis_mode() == "pv":
            return None
        if not self._use_classifier.isChecked():
            return None
        from ._learning import load_classifier

        clf = load_classifier(
            self._annot_dir.text(), filename="curated_host_features.joblib"
        )
        if clf is None:
            self._log_msg("No host classifier trained yet — batch runs without it.")
        return clf

    def _log_msg(self, msg: str):
        self._log.append(msg)


# ── napari entry point ────────────────────────────────────────────────────────


def make_batch_widget(napari_viewer=None):
    """
    Called by napari when the user opens 'Peredox: Batch Process ND2 Files'.
    """
    return BatchWidget(napari_viewer=napari_viewer)
