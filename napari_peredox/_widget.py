"""
Main dock widget for napari-peredox.

Workflow
--------
1.  User opens a multi-channel image in napari.
2.  In the 'Setup' section they assign:
      - Which layer is the image (auto-detected)
      - Which channel index = cpTSapphire
      - Which channel index = mCherry
      - Which channel to run Cellpose-SAM segmentation on (or composite)
      - Physical pixel size (µm) for area calibration — optional
      - Annotations/output directory
3.  Click 'Segment PVs':
      - Cellpose-SAM (cpsam) runs and produces a label layer in napari.
      - If a trained classifier exists in the annotations dir it is applied
        automatically as a pre-filter (false-positive removal).
      - A results table is printed to the console.
4.  Click 'Show measurements table':
      - Opens a simple in-widget table of all PVs with ratio values.
5.  Click 'Open Curation Panel':
      - Launches the CurationWidget in a new napari window.
      - User steps through each PV, accepting or rejecting.
6.  After curation, 'Save & retrain classifier' in the curation panel:
      - Appends decisions to annotations/curated_features.csv.
      - Fits a new RandomForestClassifier and saves it as curated_features.joblib.
      - Status bar updates with training-set stats.
"""

from pathlib import Path

import napari
import numpy as np
from qtpy.QtCore import QObject, Qt, QThread, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

# ── background worker ────────────────────────────────────────────────────────


class _SegmentWorker(QObject):
    """
    Runs Cellpose-SAM in a background thread so the napari UI stays responsive.
    Emits `finished` with (labels, vacuole_labels, features, measurements) when done,
    or `error` with an exception message on failure.
    """

    finished = Signal(
        object, object, object, object, object
    )  # raw_labels, labels, vac_img, features, measurements
    error = Signal(str)
    progress = Signal(str)

    def __init__(self, params: dict):
        super().__init__()
        self.params = params

    def run(self):
        try:
            p = self.params
            image = p["image"]
            ch_cptsa = p["ch_cptsa"]
            ch_mcherry = p["ch_mcherry"]
            seg_ch = p["seg_ch"]
            use_composite = p["use_composite"]
            min_area_um2 = p["min_area_um2"]
            max_area_um2 = p["max_area_um2"]
            max_eccentricity = p["max_eccentricity"]
            min_solidity = p["min_solidity"]
            pixel_size = p["pixel_size"]
            ch_names = p["ch_names"]
            classifier = p["classifier"]
            model_path = p["model_path"]
            diameter = p["diameter"]
            flow_threshold = p["flow_threshold"]
            cellprob_threshold = p["cellprob_threshold"]
            group_vacuoles = p["group_vacuoles"]
            dilation_px = p["dilation_px"]
            vacuole_method = p["vacuole_method"]
            watershed_split = p["watershed_split"]
            watershed_min_distance = p["watershed_min_distance"]
            threshold_method = p["threshold_method"]
            threshold_channel = p["threshold_channel"]
            threshold_value = p["threshold_value"]
            threshold_percentile = p["threshold_percentile"]

            thresh_str = (
                f" (threshold: {threshold_method})"
                if threshold_method != "none"
                else ""
            )
            self.progress.emit(f"Running Cellpose-SAM segmentation{thresh_str}…")

            # ---- segmentation ----
            from ._segment import apply_classifier_filter, group_by_vacuole, segment_pvs

            # Convert µm² area limits to pixel² using physical pixel size.
            # If pixel_size is 0 (unknown), disable the area filter entirely.
            if pixel_size > 0:
                min_area_px = min_area_um2 / (pixel_size**2)
                max_area_px = max_area_um2 / (pixel_size**2)
                self.progress.emit(
                    f"Area filter: {min_area_um2}–{max_area_um2} µm² "
                    f"= {min_area_px:.0f}–{max_area_px:.0f} px² "
                    f"(pixel size {pixel_size} µm/px)"
                )
            else:
                min_area_px = 0.0
                max_area_px = 1e9
                self.progress.emit(
                    "Pixel size not set — area filter disabled. "
                    "Set pixel size to enable µm² filtering."
                )

            labels, raw_labels, fstats = segment_pvs(
                image=image,
                channel_index=seg_ch,
                use_composite=use_composite,
                model_path=model_path,
                min_area_px=min_area_px,
                max_area_px=max_area_px,
                max_eccentricity=max_eccentricity,
                min_solidity=min_solidity,
                diameter=diameter,
                flow_threshold=flow_threshold,
                cellprob_threshold=cellprob_threshold,
                threshold_method=threshold_method,
                threshold_channel=threshold_channel,
                threshold_value=threshold_value,
                threshold_percentile=threshold_percentile,
                watershed_split=watershed_split,
                watershed_min_distance=watershed_min_distance,
            )

            # Log threshold diagnostic first
            if fstats["threshold_method"] != "none":
                if fstats["threshold_skipped"]:
                    self.progress.emit(
                        f"WARNING: {fstats['threshold_method'].upper()} threshold "
                        f"(cutoff={fstats['threshold_cutoff']:.1f}) removed "
                        f"{100 - fstats['pct_kept']:.1f}% of pixels — threshold "
                        f"skipped automatically to avoid blank image. "
                        f"Try 'none' or a lower percentile."
                    )
                else:
                    self.progress.emit(
                        f"Intensity threshold ({fstats['threshold_method']}, "
                        f"cutoff={fstats['threshold_cutoff']:.1f}): "
                        f"{fstats['pct_kept']:.1f}% of pixels kept."
                    )

            self.progress.emit(
                f"Cellpose found {fstats['total_raw']} segments → "
                f"{fstats['kept']} kept after morphology filter "
                f"(dropped: {fstats['rejected_area']} area, "
                f"{fstats['rejected_eccentricity']} eccentricity, "
                f"{fstats['rejected_solidity']} solidity)"
            )

            # ---- feature extraction ----
            from ._learning import extract_features

            features = extract_features(
                labels=labels,
                image=image,
                seg_channel=seg_ch,
                ch_cptsa=ch_cptsa,
                ch_mcherry=ch_mcherry,
                ch_names=ch_names,
            )

            # ---- classifier filter ----
            if classifier is not None and len(features) > 0:
                self.progress.emit("Applying trained classifier filter…")
                labels = apply_classifier_filter(labels, features, classifier)
                self.progress.emit(
                    f"After classifier filter: {labels.max()} PVs remaining."
                )
                # Re-extract features for the filtered label set
                features = extract_features(
                    labels=labels,
                    image=image,
                    seg_channel=seg_ch,
                    ch_cptsa=ch_cptsa,
                    ch_mcherry=ch_mcherry,
                    ch_names=ch_names,
                )

            # ---- measurements (per individual parasite) ----
            self.progress.emit("Computing fluorescence measurements…")
            from ._measure import measure_pvs, select_one_per_vacuole

            measurements = measure_pvs(
                labels=labels,
                image=image,
                ch_cptsa=ch_cptsa,
                ch_mcherry=ch_mcherry,
                ch_names=ch_names,
                pixel_size_um=pixel_size if pixel_size > 0 else None,
            )

            # ---- vacuole grouping: select one parasite per vacuole ----
            vac_img = None
            if group_vacuoles and not measurements.empty:
                self.progress.emit("Grouping parasites into vacuoles…")
                from ._segment import vacuole_label_image

                vacuole_map = group_by_vacuole(labels, dilation_px=dilation_px)
                vac_img = vacuole_label_image(labels, vacuole_map)
                measurements = select_one_per_vacuole(
                    measurements, vacuole_map, method=vacuole_method
                )
                n_vacuoles = measurements["vacuole_id"].nunique()
                self.progress.emit(
                    f"Grouped into {n_vacuoles} vacuoles — "
                    f"one representative parasite each ({vacuole_method})."
                )

            self.finished.emit(raw_labels, labels, vac_img, features, measurements)

        except Exception:
            import traceback

            self.error.emit(traceback.format_exc())


# ── main widget ──────────────────────────────────────────────────────────────


class PeredoxWidget(QWidget):
    """
    Main napari dock widget.  Wires together segmentation, measurement,
    curation, and the persistence layer.
    """

    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()
        self._viewer = napari_viewer

        # State set after segmentation
        self._labels: np.ndarray | None = None
        self._features = None  # pd.DataFrame
        self._measurements = None  # pd.DataFrame
        self._classifier = None  # fitted sklearn pipeline or None
        self._image_stem: str = "image"

        self._thread: QThread | None = None
        self._worker: _SegmentWorker | None = None

        self._build_ui()
        self._try_load_classifier()

    # ── UI construction ──────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(8)

        # ---- Setup group ----
        setup_box = QGroupBox("Setup")
        form = QFormLayout(setup_box)
        form.setContentsMargins(6, 6, 6, 6)

        # Layer selector
        self._layer_combo = QComboBox()
        self._layer_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._refresh_layers_btn = QPushButton("↺")
        self._refresh_layers_btn.setFixedWidth(28)
        self._refresh_layers_btn.clicked.connect(self._populate_layers)
        layer_row = QHBoxLayout()
        layer_row.addWidget(self._layer_combo, stretch=1)
        layer_row.addWidget(self._refresh_layers_btn)
        form.addRow("Image layer:", layer_row)

        # Channel spinboxes
        self._ch_cptsa = QSpinBox()
        self._ch_cptsa.setRange(0, 15)
        self._ch_cptsa.setValue(0)
        self._ch_cptsa.setToolTip("Channel index of cpTSapphire (NADH sensor)")
        form.addRow("cpTSapphire channel:", self._ch_cptsa)

        self._ch_mcherry = QSpinBox()
        self._ch_mcherry.setRange(0, 15)
        self._ch_mcherry.setValue(1)
        self._ch_mcherry.setToolTip("Channel index of mCherry (reference)")
        form.addRow("mCherry channel:", self._ch_mcherry)

        # Segmentation channel
        self._seg_ch = QSpinBox()
        self._seg_ch.setRange(0, 15)
        self._seg_ch.setValue(0)
        self._seg_ch.setToolTip(
            "Channel Cellpose-SAM uses for segmentation. "
            "Often the cpTSapphire channel works well."
        )
        self._use_composite = QCheckBox("Max composite of all channels")
        self._use_composite.setChecked(False)
        self._use_composite.setToolTip(
            "If checked, Cellpose-SAM sees the channel-wise maximum projection "
            "instead of a single channel."
        )
        seg_row = QHBoxLayout()
        seg_row.addWidget(self._seg_ch)
        seg_row.addWidget(self._use_composite)
        form.addRow("Segment on channel:", seg_row)

        # Cellpose diameter — target individual parasites, not whole vacuoles
        self._diameter = QSpinBox()
        self._diameter.setRange(0, 2000)
        self._diameter.setValue(0)
        self._diameter.setSpecialValueText("auto")
        self._diameter.setToolTip(
            "Expected parasite diameter in pixels.  0 = auto-estimate.\n"
            "Set to roughly the diameter of one parasite body (not the vacuole)."
        )
        form.addRow("Diameter (px, 0=auto):", self._diameter)

        # Vacuole grouping
        self._group_vacuoles = QCheckBox("Group parasites into vacuoles")
        self._group_vacuoles.setChecked(True)
        self._group_vacuoles.setToolTip(
            "Dilate each parasite mask and merge overlapping masks to identify\n"
            "which parasites share a vacuole.  One representative parasite per\n"
            "vacuole is then selected for reporting."
        )
        self._dilation_px = QSpinBox()
        self._dilation_px.setRange(1, 100)
        self._dilation_px.setValue(5)
        self._dilation_px.setToolTip(
            "How many pixels to expand each parasite mask when grouping.\n"
            "Increase if parasites within one vacuole are not being merged."
        )
        self._vacuole_method = QComboBox()
        self._vacuole_method.addItems(["largest", "highest_ratio", "median_ratio"])
        self._vacuole_method.setToolTip(
            "Which parasite to use as the representative for each vacuole:\n"
            "  largest       — biggest parasite (most signal, least noise)\n"
            "  highest_ratio — parasite with the highest cpTSapphire/mCherry\n"
            "  median_ratio  — parasite closest to the group median"
        )
        group_row = QHBoxLayout()
        group_row.addWidget(self._group_vacuoles)
        group_row.addWidget(QLabel("dilation:"))
        group_row.addWidget(self._dilation_px)
        form.addRow("Vacuole grouping:", group_row)
        form.addRow("Select parasite by:", self._vacuole_method)
        self._group_vacuoles.toggled.connect(self._dilation_px.setEnabled)
        self._group_vacuoles.toggled.connect(self._vacuole_method.setEnabled)

        # Watershed splitting
        self._watershed_split = QCheckBox("Split merged segments (watershed)")
        self._watershed_split.setChecked(False)
        self._watershed_split.setToolTip(
            "If Cellpose segments the whole vacuole as one object, enable this\n"
            "to split each mask into individual parasites using distance-transform\n"
            "watershed seeded by intensity peaks.\n"
            "Tune 'Min separation' to match the parasite radius in pixels."
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
        form.addRow("Watershed split:", watershed_row)
        self._watershed_split.toggled.connect(self._watershed_min_dist.setEnabled)
        self._watershed_min_dist.setEnabled(False)

        # Cellpose thresholds
        self._flow_thresh = QDoubleSpinBox()
        self._flow_thresh.setRange(0.0, 3.0)
        self._flow_thresh.setSingleStep(0.1)
        self._flow_thresh.setDecimals(2)
        self._flow_thresh.setValue(0.4)
        self._flow_thresh.setToolTip(
            "Flow error threshold.  Higher = more permissive (finds more objects).\n"
            "Lower = stricter (fewer, cleaner masks).  Default 0.4."
        )
        self._cellprob_thresh = QDoubleSpinBox()
        self._cellprob_thresh.setRange(-6.0, 6.0)
        self._cellprob_thresh.setSingleStep(0.5)
        self._cellprob_thresh.setDecimals(1)
        self._cellprob_thresh.setValue(0.0)
        self._cellprob_thresh.setToolTip(
            "Cell probability threshold.  Lower = more permissive.\n"
            "Default 0.0.  Try -1 or -2 to catch dim vacuoles."
        )
        thresh_row = QHBoxLayout()
        thresh_row.addWidget(QLabel("flow:"))
        thresh_row.addWidget(self._flow_thresh)
        thresh_row.addWidget(QLabel("prob:"))
        thresh_row.addWidget(self._cellprob_thresh)
        form.addRow("Cellpose thresholds:", thresh_row)

        # ── Pre-segmentation intensity threshold ──────────────────────────────
        # Pixels below this threshold are zeroed before cellpose runs,
        # preventing autofluorescent objects from being segmented.
        self._thresh_method = QComboBox()
        self._thresh_method.addItems(["none", "otsu", "percentile", "manual"])
        self._thresh_method.setToolTip(
            "Intensity threshold applied before segmentation.\n"
            "Pixels below the threshold are zeroed so cellpose ignores them.\n"
            "  none        — no masking\n"
            "  otsu        — automatic Otsu threshold (good starting point)\n"
            "  percentile  — threshold at Nth percentile of non-zero pixels\n"
            "  manual      — set an explicit cutoff value"
        )
        self._thresh_channel = QSpinBox()
        self._thresh_channel.setRange(0, 15)
        self._thresh_channel.setValue(0)
        self._thresh_channel.setToolTip(
            "Channel to compute the threshold on.\n"
            "Usually the same as the segmentation channel."
        )
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
            "Percentile of non-zero pixels to use as threshold.\n"
            "50 = median; increase to be more aggressive."
        )
        thresh_method_row = QHBoxLayout()
        thresh_method_row.addWidget(self._thresh_method)
        thresh_method_row.addWidget(QLabel("ch:"))
        thresh_method_row.addWidget(self._thresh_channel)
        form.addRow("Intensity threshold:", thresh_method_row)
        thresh_val_row = QHBoxLayout()
        thresh_val_row.addWidget(QLabel("value:"))
        thresh_val_row.addWidget(self._thresh_value)
        thresh_val_row.addWidget(QLabel("pct:"))
        thresh_val_row.addWidget(self._thresh_percentile)
        form.addRow("", thresh_val_row)

        def _update_thresh_controls(method: str):
            self._thresh_value.setEnabled(method == "manual")
            self._thresh_percentile.setEnabled(method == "percentile")
            self._thresh_channel.setEnabled(method != "none")

        self._thresh_method.currentTextChanged.connect(_update_thresh_controls)
        _update_thresh_controls("none")  # set initial state

        # Morphology filter — size (area µm²), roundness (eccentricity), solidity
        self._min_area_um2 = QDoubleSpinBox()
        self._min_area_um2.setRange(0.0, 100_000.0)
        self._min_area_um2.setDecimals(1)
        self._min_area_um2.setValue(5.0)
        self._min_area_um2.setToolTip(
            "Minimum parasite area in µm². Smaller objects are discarded.\n"
            "Typical parasites at 60× with 0.105 µm/px: 5–25 µm².\n"
            "Converted to pixels using the pixel size above. Set to 0 to disable."
        )
        self._max_area_um2 = QDoubleSpinBox()
        self._max_area_um2.setRange(0.0, 100_000.0)
        self._max_area_um2.setDecimals(1)
        self._max_area_um2.setValue(25.0)
        self._max_area_um2.setToolTip(
            "Maximum parasite area in µm². Larger objects are discarded.\n"
            "Typical parasites at 60× with 0.105 µm/px: 5–25 µm².\n"
            "Converted to pixels using the pixel size above. Set to 100000 to disable."
        )
        area_row = QHBoxLayout()
        area_row.addWidget(QLabel("min:"))
        area_row.addWidget(self._min_area_um2)
        area_row.addWidget(QLabel("max:"))
        area_row.addWidget(self._max_area_um2)
        form.addRow("Area filter (µm²):", area_row)

        self._max_eccentricity = QDoubleSpinBox()
        self._max_eccentricity.setRange(0.0, 1.0)
        self._max_eccentricity.setSingleStep(0.05)
        self._max_eccentricity.setDecimals(2)
        self._max_eccentricity.setValue(0.95)
        self._max_eccentricity.setToolTip(
            "Reject segments more elongated than this.\n"
            "0 = perfect circle, 1 = line. Parasites are typically < 0.95.\n"
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
        form.addRow("Shape filter:", shape_row)

        # Pixel size
        self._pixel_size = QDoubleSpinBox()
        self._pixel_size.setRange(0, 100)
        self._pixel_size.setDecimals(4)
        self._pixel_size.setValue(0.0)
        self._pixel_size.setToolTip(
            "Physical pixel size in µm. Leave at 0 to skip area calibration."
        )
        form.addRow("Pixel size (µm):", self._pixel_size)

        # Annotations directory
        self._annot_dir = QLineEdit()
        default_dir = str(Path(__file__).parent.parent / "annotations")
        self._annot_dir.setText(default_dir)
        browse_btn = QPushButton("…")
        browse_btn.setFixedWidth(28)
        browse_btn.clicked.connect(self._browse_annot_dir)
        dir_row = QHBoxLayout()
        dir_row.addWidget(self._annot_dir, stretch=1)
        dir_row.addWidget(browse_btn)
        form.addRow("Annotations dir:", dir_row)

        root.addWidget(setup_box)

        # ---- Action buttons ----
        self._btn_segment = QPushButton("▶ Segment PVs")
        self._btn_segment.setStyleSheet("font-weight: bold;")
        self._btn_segment.clicked.connect(self._run_segmentation)
        root.addWidget(self._btn_segment)

        self._btn_table = QPushButton("📊 Show measurements table")
        self._btn_table.clicked.connect(self._show_table)
        self._btn_table.setEnabled(False)
        root.addWidget(self._btn_table)

        self._btn_curate = QPushButton("🔍 Open Curation Panel")
        self._btn_curate.clicked.connect(self._open_curation)
        self._btn_curate.setEnabled(False)
        root.addWidget(self._btn_curate)

        self._btn_save_csv = QPushButton("💾 Export measurements CSV")
        self._btn_save_csv.clicked.connect(self._export_csv)
        self._btn_save_csv.setEnabled(False)
        root.addWidget(self._btn_save_csv)

        # ---- Classifier status ----
        clf_box = QGroupBox("Classifier status")
        clf_layout = QVBoxLayout(clf_box)
        self._use_classifier = QCheckBox("Apply classifier filter at segmentation")
        self._use_classifier.setChecked(False)
        self._use_classifier.setToolTip(
            "When checked, a trained RandomForest classifier automatically removes\n"
            "false-positive segments before measurements are computed.\n"
            "Uncheck until you have enough curated data for a reliable model."
        )
        clf_layout.addWidget(self._use_classifier)
        self._clf_status = QLabel("No classifier trained yet.")
        self._clf_status.setWordWrap(True)
        clf_layout.addWidget(self._clf_status)
        root.addWidget(clf_box)

        # ---- Log ----
        log_box = QGroupBox("Log")
        log_layout = QVBoxLayout(log_box)
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(120)
        log_layout.addWidget(self._log)
        root.addWidget(log_box)

        root.addStretch()

        # Populate layer list immediately and wire updates
        self._populate_layers()
        self._viewer.layers.events.inserted.connect(lambda _: self._populate_layers())
        self._viewer.layers.events.removed.connect(lambda _: self._populate_layers())
        self._layer_combo.currentTextChanged.connect(
            lambda _: self._autofill_pixel_size()
        )

    # ── helpers ──────────────────────────────────────────────────────────────

    def _log_msg(self, msg: str):
        self._log.append(msg)

    def _populate_layers(self):
        """Refresh the layer dropdown to show current napari image layers."""
        from napari.layers import Image

        current = self._layer_combo.currentText()
        self._layer_combo.clear()
        for layer in self._viewer.layers:
            if isinstance(layer, Image):
                self._layer_combo.addItem(layer.name)
        # Restore previous selection if it still exists
        idx = self._layer_combo.findText(current)
        if idx >= 0:
            self._layer_combo.setCurrentIndex(idx)
        self._autofill_pixel_size()

    def _autofill_pixel_size(self):
        """
        Populate the pixel-size field from the selected layer's scale metadata
        if the field is currently 0 (i.e. the user has not set it manually).

        napari stores physical scale as layer.scale — a tuple of µm/pixel values
        for each axis.  For a 2-D image it is (y_scale, x_scale); for a 3-D
        (C, H, W) layer it is (1.0, y_scale, x_scale).  We take the last
        axis value as the in-plane pixel size.

        ND2 and OME-TIFF files opened via napari-nd2 / tifffile carry their
        calibration in layer.scale automatically.  Plain TIFFs typically have
        scale = (1.0, 1.0) and will not auto-populate.
        """
        if self._pixel_size.value() > 0:
            return  # user has already set a value — don't overwrite

        from napari.layers import Image

        name = self._layer_combo.currentText()
        if not name or name not in self._viewer.layers:
            return
        layer = self._viewer.layers[name]
        if not isinstance(layer, Image):
            return

        scale = layer.scale  # tuple, one entry per axis
        # In-plane pixel size is always the last two axes (y, x); take x.
        px = float(scale[-1]) if len(scale) >= 1 else 0.0
        # Only accept plausible values (0.01–100 µm/px); ignore default 1.0
        # unless it looks genuinely calibrated (not the napari default of all-ones).
        if 0.01 < px < 100.0 and not all(s == 1.0 for s in scale):
            self._pixel_size.setValue(round(px, 4))
            self._log_msg(f"Pixel size auto-read from layer scale: {px:.4f} µm/px")

    def _get_image_array(self) -> np.ndarray:
        """
        Return the selected image as a (H, W, C) float32 numpy array.
        Handles 2-D (H, W), 3-D (C, H, W) napari format, and already-(H,W,C).
        """
        name = self._layer_combo.currentText()
        if not name:
            raise RuntimeError("No image layer selected.")
        layer = self._viewer.layers[name]
        data = np.asarray(layer.data).astype(np.float32)

        if data.ndim == 2:
            # Single channel — expand to (H, W, 1)
            return data[..., np.newaxis]
        elif data.ndim == 3:
            # napari stores multi-channel as (C, H, W); convert to (H, W, C)
            if data.shape[0] < data.shape[1] and data.shape[0] < data.shape[2]:
                return np.moveaxis(data, 0, -1)
            return data
        elif data.ndim == 4:
            # (Z, C, H, W) or (Z, H, W, C) — take middle Z slice
            mid = data.shape[0] // 2
            plane = data[mid]
            if plane.shape[0] < plane.shape[1]:
                return np.moveaxis(plane, 0, -1)
            return plane.astype(np.float32)
        else:
            raise RuntimeError(f"Unsupported image dimensionality: {data.shape}")

    def _browse_annot_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select annotations directory")
        if d:
            self._annot_dir.setText(d)

    def _try_load_classifier(self):
        """Attempt to load a previously trained classifier from the annotations dir."""
        from ._learning import load_classifier

        annot_dir = self._annot_dir.text()
        clf = load_classifier(annot_dir)
        self._classifier = clf
        self._update_clf_status()

    def _update_clf_status(self):
        from ._learning import classifier_stats

        annot_dir = self._annot_dir.text()
        csv_path = Path(annot_dir) / "curated_features.csv"
        stats = classifier_stats(csv_path)
        if self._classifier is not None:
            self._clf_status.setText(
                f"Trained classifier loaded.\n"
                f"Training set: {stats['total']} PVs "
                f"({stats['accepted']} accepted, {stats['rejected']} rejected).\n"
                f"Will be applied automatically at segmentation."
            )
        elif stats["total"] > 0:
            self._clf_status.setText(
                f"{stats['total']} annotations on disk but classifier not yet trained.\n"
                f"Run a curation session and click 'Save & retrain'."
            )
        else:
            self._clf_status.setText(
                "No classifier trained yet.\n"
                "Curate PVs to start building training data."
            )

    # ── segmentation ─────────────────────────────────────────────────────────

    def _run_segmentation(self):
        """Collect parameters and launch the background segmentation worker."""
        try:
            image = self._get_image_array()
        except RuntimeError as exc:
            self._log_msg(f"Error: {exc}")
            return

        n_ch = image.shape[-1]
        ch_names = {i: f"ch{i}" for i in range(n_ch)}
        ch_names[self._ch_cptsa.value()] = "cptsa"
        ch_names[self._ch_mcherry.value()] = "mcherry"

        # Derive a stem name from the selected layer
        self._image_stem = self._layer_combo.currentText().replace(" ", "_") or "image"

        # If pixel size is not set, ask the user before continuing.
        # Without it the area filter (µm²) cannot be applied.
        if self._pixel_size.value() == 0:
            val, ok = QInputDialog.getDouble(
                self,
                "Pixel size not set",
                "Enter physical pixel size (µm/px).\n"
                "Required for the µm² area filter.\n"
                "Cancel or enter 0 to skip (area filter will be disabled):",
                decimals=4,
                min=0.0,
                max=100.0,
                value=0.105,
            )
            if ok and val > 0:
                self._pixel_size.setValue(val)
                self._log_msg(f"Pixel size set to {val:.4f} µm/px.")

        diameter = self._diameter.value()
        params = {
            "image": image,
            "ch_cptsa": self._ch_cptsa.value(),
            "ch_mcherry": self._ch_mcherry.value(),
            "seg_ch": self._seg_ch.value(),
            "use_composite": self._use_composite.isChecked(),
            "min_area_um2": self._min_area_um2.value(),
            "max_area_um2": self._max_area_um2.value(),
            "max_eccentricity": self._max_eccentricity.value(),
            "min_solidity": self._min_solidity.value(),
            "pixel_size": self._pixel_size.value(),
            "ch_names": ch_names,
            "classifier": (
                self._classifier if self._use_classifier.isChecked() else None
            ),
            "model_path": None,
            "diameter": diameter if diameter > 0 else None,
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

        self._btn_segment.setEnabled(False)
        self._btn_segment.setText("Running…")

        # Pre-load the model in the main thread so the CUDA context is
        # established before the worker starts, and the device appears in the log.
        from ._segment import preload_model

        self._log_msg(preload_model())
        self._log_msg("Starting segmentation…")

        # Run in a background thread to keep the UI alive
        self._thread = QThread()
        self._worker = _SegmentWorker(params)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_segmentation_done)
        self._worker.error.connect(self._on_segmentation_error)
        self._worker.progress.connect(self._log_msg)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)

        self._thread.start()

    def _on_segmentation_done(
        self, raw_labels, labels, vac_img, features, measurements
    ):
        """Called in the main thread when the worker finishes successfully."""
        self._labels = labels
        self._features = features
        self._measurements = measurements

        stem = self._image_stem

        # ── Raw Cellpose candidates (before morphology filter) ───────────────
        # Shown dimmed so the user can immediately see what Cellpose detected
        # vs what survived the shape filter.
        raw_name = f"{stem}_parasite_candidates"
        if raw_name in self._viewer.layers:
            self._viewer.layers[raw_name].data = raw_labels
        else:
            self._viewer.add_labels(raw_labels, name=raw_name, opacity=0.25)

        # ── Filtered parasite labels ─────────────────────────────────────────
        parasite_name = f"{stem}_parasites"
        if parasite_name in self._viewer.layers:
            self._viewer.layers[parasite_name].data = labels
        else:
            self._viewer.add_labels(labels, name=parasite_name)

        # ── Vacuole grouping layer ────────────────────────────────────────────
        vac_name = f"{stem}_vacuoles"
        if vac_img is not None:
            if vac_name in self._viewer.layers:
                self._viewer.layers[vac_name].data = vac_img
            else:
                self._viewer.add_labels(vac_img, name=vac_name, opacity=0.4)
        elif vac_name in self._viewer.layers:
            self._viewer.layers.remove(vac_name)

        n_parasites = int(labels.max())
        n_vacuoles = int(vac_img.max()) if vac_img is not None else n_parasites
        self._log_msg(
            f"Done — {n_parasites} parasites in {n_vacuoles} vacuoles. "
            f"({raw_labels.max()} raw Cellpose segments; toggle '{raw_name}' "
            f"layer to see what was filtered out.)"
        )

        self._btn_segment.setEnabled(True)
        self._btn_segment.setText("▶ Segment PVs")
        self._btn_table.setEnabled(True)
        self._btn_curate.setEnabled(True)
        self._btn_save_csv.setEnabled(True)

    def _on_segmentation_error(self, msg: str):
        self._log_msg(f"Segmentation error:\n{msg}")
        self._btn_segment.setEnabled(True)
        self._btn_segment.setText("▶ Segment PVs")

    # ── measurement table ────────────────────────────────────────────────────

    def _show_table(self):
        """Display the measurements DataFrame in a simple dock table."""
        if self._measurements is None or len(self._measurements) == 0:
            self._log_msg("No measurements available.")
            return

        df = self._measurements.reset_index()
        table_win = QWidget(self, Qt.Window)
        table_win.setWindowTitle("PV Measurements")
        table_win.resize(900, 400)
        layout = QVBoxLayout(table_win)

        table = QTableWidget(len(df), len(df.columns))
        table.setHorizontalHeaderLabels(df.columns.tolist())
        for row_idx, row in df.iterrows():
            for col_idx, val in enumerate(row):
                item = QTableWidgetItem(
                    f"{val:.4f}" if isinstance(val, float) else str(val)
                )
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                table.setItem(row_idx, col_idx, item)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        layout.addWidget(table)
        table_win.show()

    # ── CSV export ───────────────────────────────────────────────────────────

    def _export_csv(self):
        """Save measurements and label image to the annotations/results dir."""
        from ._io import save_labels, save_measurements

        if self._measurements is None:
            return
        annot_dir = self._annot_dir.text()
        meas_path = save_measurements(self._measurements, self._image_stem, annot_dir)
        if self._labels is not None:
            save_labels(self._labels, self._image_stem, annot_dir)
        self._log_msg(f"Saved → {meas_path}")

    # ── curation ─────────────────────────────────────────────────────────────

    def _open_curation(self):
        """Launch the CurationWidget as a floating window."""
        if self._labels is None or self._measurements is None:
            self._log_msg("Run segmentation first.")
            return

        # Retrieve the original image for thumbnails
        try:
            image = self._get_image_array()
        except RuntimeError:
            image = np.zeros((*self._labels.shape, 2), dtype=np.float32)

        from ._curation import CurationWidget

        n_ch = image.shape[-1]
        ch_names = {i: f"ch{i}" for i in range(n_ch)}
        ch_names[self._ch_cptsa.value()] = "cptsa"
        ch_names[self._ch_mcherry.value()] = "mcherry"
        px = self._pixel_size.value()

        self._curation_win = CurationWidget(
            labels=self._labels,
            image=image,
            measurements=self._measurements,
            ch_cptsa=self._ch_cptsa.value(),
            ch_mcherry=self._ch_mcherry.value(),
            ch_names=ch_names,
            pixel_size_um=px if px > 0 else None,
            on_save=self._on_curation_saved,
            parent=None,
        )
        self._curation_win.setWindowTitle("Peredox — PV Curation")
        self._curation_win.resize(320, 520)
        self._curation_win.show()

    def _on_curation_saved(
        self, decisions: dict, vacuole_assignments: dict | None = None
    ):
        """
        Called by CurationWidget when the user clicks 'Save & retrain'.
        Appends annotations, re-trains the classifier, updates status.
        """
        from ._io import append_curated_annotations
        from ._learning import train_classifier

        annot_dir = self._annot_dir.text()
        csv_path = append_curated_annotations(
            decisions=decisions,
            features=self._features if self._features is not None else {},
            image_stem=self._image_stem,
            annotations_dir=annot_dir,
            vacuole_assignments=vacuole_assignments,
        )
        n_decided = sum(1 for v in decisions.values() if v in (0, 1))
        self._log_msg(f"Saved {n_decided} annotations → {csv_path}")

        # Retrain
        clf = train_classifier(csv_path)
        if clf is not None:
            self._classifier = clf
            self._log_msg("Classifier retrained successfully.")
        else:
            self._log_msg(
                "Not enough labelled data to train yet "
                "(need ≥10 examples with both classes)."
            )
        self._update_clf_status()


# ── napari entry point ────────────────────────────────────────────────────────


def make_main_widget(napari_viewer: napari.Viewer) -> PeredoxWidget:
    """Called by napari when the user opens 'Peredox: Segment & Measure PVs'."""
    return PeredoxWidget(napari_viewer)
