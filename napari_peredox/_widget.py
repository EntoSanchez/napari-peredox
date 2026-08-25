"""
Main dock widget for napari-peredox  —  two-stage pipeline.

Tab layout
----------
Setup      Image layer, channels, pixel size, annotations dir,
           filter params, cpSAM tuning.
Segment    Stage 1 (vacuoles) → review → Stage 2 (parasites) → review →
           export.  Log panel.
Training   Pair counts, clear buttons, load previous results.

Two-stage workflow
------------------
Stage 1  StarDist vacuole model (or cpSAM fallback) detects whole PVs.
         User reviews in VacuoleCurationWidget — accept/reject/redraw.
         Saving writes vacuole training pair + triggers vacuole model
         retraining.

Stage 2  StarDist parasite model (or cpSAM fallback) detects individual
         parasites inside each accepted vacuole crop.  Parasite labels are
         clipped to the vacuole mask; non-overlapping assignment enforced.
         User reviews in CurationWidget (per-vacuole gallery).
         Saving writes parasite training pair + triggers parasite model
         retraining.
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
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

# ---------------------------------------------------------------------------
# Background worker — Stage 1: vacuole segmentation
# ---------------------------------------------------------------------------


class _VacuoleWorker(QObject):
    """Run vacuole segmentation in a background thread."""

    finished = Signal(object, object)  # (raw_vac_labels, vac_labels)
    error = Signal(str)
    progress = Signal(str)

    def __init__(self, params: dict):
        super().__init__()
        self.params = params

    def run(self) -> None:
        try:
            p = self.params
            image = p["image"]
            seg_ch = p["seg_ch"]
            seg_backend = p.get("seg_backend", 0)
            annot_dir = p.get("annot_dir", "")
            min_area_px = p["min_area_px"]
            max_area_px = p["max_area_px"]
            max_eccentricity = p["max_eccentricity"]
            min_solidity = p["min_solidity"]

            if seg_backend == 1:
                # StarDist vacuole model
                from ._segment import filter_labels
                from ._stardist import load_stardist_model, predict_stardist

                model_dir = Path(annot_dir) / "stardist_model"
                self.progress.emit("Loading vacuole StarDist model…")
                sd_model = load_stardist_model(model_dir, mode="vacuoles")
                if sd_model is None:
                    raise RuntimeError(
                        "Vacuole StarDist model not found — train it first "
                        "or use cpSAM fallback."
                    )
                self.progress.emit("Running StarDist vacuole inference…")
                raw_labels = predict_stardist(image, sd_model, seg_ch)
                vac_labels, raw_out, fstats = filter_labels(
                    raw_labels,
                    min_area_px,
                    max_area_px,
                    max_eccentricity,
                    min_solidity,
                )
                self.progress.emit(
                    f"StarDist: {fstats['total_raw']} vacuoles → "
                    f"{fstats['kept']} after filter."
                )
            else:
                # cpSAM fallback for vacuole detection
                from ._segment import segment_pvs

                use_composite = p.get("use_composite", False)
                diameter = p.get("diameter", None)
                flow_threshold = p.get("flow_threshold", 0.4)
                cellprob_threshold = p.get("cellprob_threshold", 0.0)
                threshold_method = p.get("threshold_method", "none")
                threshold_channel = p.get("threshold_channel", 0)
                threshold_value = p.get("threshold_value", 0.0)
                threshold_percentile = p.get("threshold_percentile", 50.0)
                watershed_split = p.get("watershed_split", False)
                watershed_min_distance = p.get("watershed_min_distance", 10)

                self.progress.emit("Running cpSAM for vacuole detection…")
                vac_labels, raw_out, fstats = segment_pvs(
                    image=image,
                    channel_index=seg_ch,
                    use_composite=use_composite,
                    model_path=None,
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
                self.progress.emit(
                    f"cpSAM: {fstats['total_raw']} vacuoles → "
                    f"{fstats['kept']} after filter."
                )

            self.finished.emit(raw_out, vac_labels)

        except Exception:
            import traceback

            self.error.emit(traceback.format_exc())


# ---------------------------------------------------------------------------
# Background worker — Stage 2: parasite segmentation within vacuoles
# ---------------------------------------------------------------------------


class _ParasiteWorker(QObject):
    """Run per-vacuole parasite segmentation in a background thread."""

    finished = Signal(object, object, object, object)
    # (para_labels, vacuole_map, features, measurements)
    error = Signal(str)
    progress = Signal(str)

    def __init__(self, params: dict):
        super().__init__()
        self.params = params

    def run(self) -> None:
        try:
            p = self.params
            image = p["image"]
            vac_labels = p["vac_labels"]
            seg_ch = p["seg_ch"]
            seg_backend = p.get("seg_backend", 0)
            annot_dir = p.get("annot_dir", "")
            ch_cptsa = p["ch_cptsa"]
            ch_mcherry = p["ch_mcherry"]
            ch_names = p["ch_names"]
            pixel_size = p.get("pixel_size", 0.0)
            min_area_px = p["min_area_px"]
            max_area_px = p["max_area_px"]
            max_eccentricity = p["max_eccentricity"]
            min_solidity = p["min_solidity"]
            vacuole_method = p.get("vacuole_method", "largest")

            # Load parasite model (or None for cpSAM fallback)
            sd_model = None
            if seg_backend == 1:
                from ._stardist import load_stardist_model

                model_dir = Path(annot_dir) / "stardist_model"
                self.progress.emit("Loading parasite StarDist model…")
                sd_model = load_stardist_model(model_dir, mode="parasites")
                if sd_model is None:
                    self.progress.emit(
                        "Parasite StarDist model not found — using cpSAM fallback."
                    )

            from ._segment import segment_parasites_in_vacuoles

            para_labels, vacuole_map = segment_parasites_in_vacuoles(
                image=image,
                vac_labels=vac_labels,
                seg_channel=seg_ch,
                model=sd_model,
                min_area_px=min_area_px,
                max_area_px=max_area_px,
                max_eccentricity=max_eccentricity,
                min_solidity=min_solidity,
                progress_cb=self.progress.emit,
            )

            # Feature extraction for classifier
            from ._learning import extract_features

            features = extract_features(
                labels=para_labels,
                image=image,
                seg_channel=seg_ch,
                ch_cptsa=ch_cptsa,
                ch_mcherry=ch_mcherry,
                ch_names=ch_names,
            )

            # Measurements
            from ._measure import (
                add_estimated_parasites,
                measure_pvs,
                measure_vacuoles,
                select_one_per_vacuole,
            )

            self.progress.emit("Computing per-parasite measurements…")
            measurements = measure_pvs(
                labels=para_labels,
                image=image,
                ch_cptsa=ch_cptsa,
                ch_mcherry=ch_mcherry,
                ch_names=ch_names,
                pixel_size_um=pixel_size if pixel_size > 0 else None,
            )

            if not measurements.empty:
                measurements = select_one_per_vacuole(
                    measurements, vacuole_map, method=vacuole_method
                )
                pv_meas = measure_vacuoles(
                    labels=para_labels,
                    image=image,
                    vacuole_map=vacuole_map,
                    ch_cptsa=ch_cptsa,
                    ch_mcherry=ch_mcherry,
                    ch_names=ch_names,
                    pixel_size_um=pixel_size if pixel_size > 0 else None,
                )
                if not pv_meas.empty:
                    pv_meas = pv_meas.rename(
                        columns={c: f"pv_{c}" for c in pv_meas.columns}
                    )
                    pv_meas.index.name = "vacuole_id"
                    measurements = measurements.join(
                        pv_meas, on="vacuole_id", how="left"
                    )
                measurements = add_estimated_parasites(measurements)

            n_para = int(para_labels.max())
            n_vac = len(set(vacuole_map.values()))
            self.progress.emit(f"Stage 2 done: {n_para} parasites in {n_vac} vacuoles.")
            self.finished.emit(para_labels, vacuole_map, features, measurements)

        except Exception:
            import traceback

            self.error.emit(traceback.format_exc())


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


# ---------------------------------------------------------------------------
# Main widget
# ---------------------------------------------------------------------------


class PeredoxWidget(QWidget):
    """Main napari dock widget — two-stage vacuole+parasite pipeline."""

    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()
        self._viewer = napari_viewer

        # Stage 1 state
        self._vac_labels: np.ndarray | None = None
        self._image_stem: str = "image"

        # Stage 2 state
        self._para_labels: np.ndarray | None = None
        self._features = None
        self._measurements = None
        self._vac_map: dict = {}

        # Host-mode state (Host tab) — never shared with PV-mode state above
        self._host_labels: np.ndarray | None = None
        self._host_para_labels: np.ndarray | None = None
        self._host_vac_map: dict = {}
        self._host_features = None
        self._host_para_measurements = None
        self._host_measurements = None
        self._host_classifier = None
        self._host_thread: QThread | None = None
        self._host_worker = None
        self._host_review_saved: bool = False

        self._classifier = None
        self._thread: QThread | None = None
        self._worker = None

        self._build_ui()
        self._try_load_classifier()

    # ── UI construction ──────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.setMinimumWidth(400)
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        tabs = QTabWidget()
        tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        tabs.addTab(self._build_setup_tab(), "Setup")
        tabs.addTab(self._build_segment_tab(), "Segment")
        tabs.addTab(self._build_host_tab(), "Host")
        tabs.addTab(self._build_training_tab(), "Training")

        root.addWidget(tabs)

    # ── Setup tab ────────────────────────────────────────────────────────────

    def _build_setup_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        # ---- Image & channels ----
        img_box = QGroupBox("Image & channels")
        form = QFormLayout(img_box)
        form.setContentsMargins(6, 6, 6, 6)

        self._layer_combo = QComboBox()
        self._layer_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._refresh_layers_btn = QPushButton("↺")
        self._refresh_layers_btn.setFixedWidth(26)
        self._refresh_layers_btn.clicked.connect(self._populate_layers)
        layer_row = QHBoxLayout()
        layer_row.addWidget(self._layer_combo, stretch=1)
        layer_row.addWidget(self._refresh_layers_btn)
        form.addRow("Image layer:", layer_row)

        self._ch_cptsa = QSpinBox()
        self._ch_cptsa.setRange(0, 15)
        self._ch_cptsa.setValue(0)
        self._ch_cptsa.setToolTip("Channel index of cpTSapphire (NADH sensor)")
        form.addRow("cpTSapphire ch:", self._ch_cptsa)

        self._ch_mcherry = QSpinBox()
        self._ch_mcherry.setRange(0, 15)
        self._ch_mcherry.setValue(1)
        self._ch_mcherry.setToolTip("Channel index of mCherry (reference)")
        form.addRow("mCherry ch:", self._ch_mcherry)

        self._seg_ch = QSpinBox()
        self._seg_ch.setRange(0, 15)
        self._seg_ch.setValue(0)
        self._seg_ch.setToolTip("Channel used as input for segmentation")
        form.addRow("Segment on ch:", self._seg_ch)

        self._pixel_size = QDoubleSpinBox()
        self._pixel_size.setRange(0, 100)
        self._pixel_size.setDecimals(4)
        self._pixel_size.setValue(0.0)
        self._pixel_size.setToolTip(
            "Physical pixel size in µm. Leave 0 to skip area calibration."
        )
        form.addRow("Pixel size (µm):", self._pixel_size)

        self._annot_dir = QLineEdit()
        default_dir = str(Path(__file__).parent.parent / "annotations")
        self._annot_dir.setText(default_dir)
        browse_btn = QPushButton("…")
        browse_btn.setFixedWidth(26)
        browse_btn.clicked.connect(self._browse_annot_dir)
        dir_row = QHBoxLayout()
        dir_row.addWidget(self._annot_dir, stretch=1)
        dir_row.addWidget(browse_btn)
        form.addRow("Annotations dir:", dir_row)

        layout.addWidget(img_box)

        # ---- Segmentation backend ----
        seg_box = QGroupBox("Segmentation")
        seg_form = QFormLayout(seg_box)
        seg_form.setContentsMargins(6, 6, 6, 6)

        self._seg_backend = QComboBox()
        self._seg_backend.addItems(["cpSAM (Cellpose)", "StarDist (fine-tuned)"])
        self._seg_backend.setToolTip(
            "cpSAM — Cellpose-SAM, works without training data.\n"
            "StarDist — your fine-tuned models (train first in Training tab)."
        )
        seg_form.addRow("Model:", self._seg_backend)

        self._use_composite = QCheckBox("Max composite of all channels")
        self._use_composite.setToolTip(
            "Cellpose-SAM sees the per-pixel channel maximum."
        )
        seg_form.addRow("", self._use_composite)

        self._diameter = QSpinBox()
        self._diameter.setRange(0, 2000)
        self._diameter.setValue(0)
        self._diameter.setSpecialValueText("auto")
        self._diameter.setToolTip(
            "Expected object diameter in pixels (0 = auto-estimate)."
        )
        seg_form.addRow("Diameter (px):", self._diameter)

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
        cp_row = QHBoxLayout()
        cp_row.addWidget(QLabel("flow:"))
        cp_row.addWidget(self._flow_thresh)
        cp_row.addWidget(QLabel("prob:"))
        cp_row.addWidget(self._cellprob_thresh)
        seg_form.addRow("cpSAM thresholds:", cp_row)

        self._watershed_split = QCheckBox("Watershed split")
        self._watershed_split.setChecked(False)
        self._watershed_min_dist = QSpinBox()
        self._watershed_min_dist.setRange(2, 200)
        self._watershed_min_dist.setValue(10)
        self._watershed_min_dist.setEnabled(False)
        ws_row = QHBoxLayout()
        ws_row.addWidget(self._watershed_split)
        ws_row.addWidget(QLabel("min sep:"))
        ws_row.addWidget(self._watershed_min_dist)
        seg_form.addRow("", ws_row)
        self._watershed_split.toggled.connect(self._watershed_min_dist.setEnabled)

        # Intensity threshold
        self._thresh_method = QComboBox()
        self._thresh_method.addItems(["none", "otsu", "percentile", "manual"])
        self._thresh_channel = QSpinBox()
        self._thresh_channel.setRange(0, 15)
        self._thresh_channel.setEnabled(False)
        self._thresh_value = QDoubleSpinBox()
        self._thresh_value.setRange(0.0, 1e9)
        self._thresh_value.setDecimals(1)
        self._thresh_value.setEnabled(False)
        self._thresh_percentile = QDoubleSpinBox()
        self._thresh_percentile.setRange(0.0, 100.0)
        self._thresh_percentile.setSingleStep(5.0)
        self._thresh_percentile.setDecimals(1)
        self._thresh_percentile.setValue(50.0)
        self._thresh_percentile.setEnabled(False)
        thr_row1 = QHBoxLayout()
        thr_row1.addWidget(self._thresh_method)
        thr_row1.addWidget(QLabel("ch:"))
        thr_row1.addWidget(self._thresh_channel)
        seg_form.addRow("Intensity threshold:", thr_row1)
        thr_row2 = QHBoxLayout()
        thr_row2.addWidget(QLabel("val:"))
        thr_row2.addWidget(self._thresh_value)
        thr_row2.addWidget(QLabel("pct:"))
        thr_row2.addWidget(self._thresh_percentile)
        seg_form.addRow("", thr_row2)

        def _upd_thresh(method: str) -> None:
            self._thresh_value.setEnabled(method == "manual")
            self._thresh_percentile.setEnabled(method == "percentile")
            self._thresh_channel.setEnabled(method != "none")

        self._thresh_method.currentTextChanged.connect(_upd_thresh)
        _upd_thresh("none")

        # Disable cpSAM controls when StarDist selected
        self._cpsam_only = [
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

        def _on_backend_changed(_: int) -> None:
            is_cpsam = self._seg_backend.currentIndex() == 0
            for ww in self._cpsam_only:
                ww.setEnabled(is_cpsam)

        self._seg_backend.currentIndexChanged.connect(_on_backend_changed)
        layout.addWidget(seg_box)

        # ---- Morphology filters ----
        filt_box = QGroupBox("Morphology filters")
        filt_form = QFormLayout(filt_box)
        filt_form.setContentsMargins(6, 6, 6, 6)

        self._min_area_um2 = QDoubleSpinBox()
        self._min_area_um2.setRange(0.0, 1e5)
        self._min_area_um2.setDecimals(1)
        self._min_area_um2.setValue(5.0)
        self._max_area_um2 = QDoubleSpinBox()
        self._max_area_um2.setRange(0.0, 1e5)
        self._max_area_um2.setDecimals(1)
        self._max_area_um2.setValue(200.0)
        area_row = QHBoxLayout()
        area_row.addWidget(QLabel("min:"))
        area_row.addWidget(self._min_area_um2)
        area_row.addWidget(QLabel("max:"))
        area_row.addWidget(self._max_area_um2)
        filt_form.addRow("Area (µm²):", area_row)

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

        self._vacuole_method = QComboBox()
        self._vacuole_method.addItems(
            ["largest", "highest_ratio", "median_ratio", "mean_ratio"]
        )
        self._vacuole_method.setToolTip(
            "Which parasite to report as the representative for each vacuole."
        )
        filt_form.addRow("Representative:", self._vacuole_method)

        self._use_classifier = QCheckBox("Apply classifier filter")
        self._use_classifier.setToolTip(
            "Use the trained RandomForest to remove false positives."
        )
        filt_form.addRow("", self._use_classifier)

        layout.addWidget(filt_box)
        layout.addStretch()

        # Wire up layer events
        self._populate_layers()
        self._viewer.layers.events.inserted.connect(lambda _: self._populate_layers())
        self._viewer.layers.events.removed.connect(lambda _: self._populate_layers())
        self._layer_combo.currentTextChanged.connect(
            lambda _: self._autofill_pixel_size()
        )

        return w

    # ── Segment tab ──────────────────────────────────────────────────────────

    def _build_segment_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # Stage 1
        s1_box = QGroupBox("Stage 1 — Vacuole detection")
        s1_layout = QVBoxLayout(s1_box)
        self._btn_stage1 = QPushButton("▶ Detect vacuoles")
        self._btn_stage1.setStyleSheet("font-weight: bold;")
        self._btn_stage1.clicked.connect(self._run_stage1)
        s1_layout.addWidget(self._btn_stage1)
        self._lbl_stage1 = QLabel("Not run yet.")
        self._lbl_stage1.setWordWrap(True)
        s1_layout.addWidget(self._lbl_stage1)
        self._btn_review_vac = QPushButton("🔍 Review vacuoles")
        self._btn_review_vac.clicked.connect(self._open_vacuole_curation)
        self._btn_review_vac.setEnabled(False)
        s1_layout.addWidget(self._btn_review_vac)
        layout.addWidget(s1_box)

        # Stage 2
        s2_box = QGroupBox("Stage 2 — Parasite detection")
        s2_layout = QVBoxLayout(s2_box)
        self._btn_stage2 = QPushButton("▶ Detect parasites in vacuoles")
        self._btn_stage2.setStyleSheet("font-weight: bold;")
        self._btn_stage2.clicked.connect(self._run_stage2)
        self._btn_stage2.setEnabled(False)
        s2_layout.addWidget(self._btn_stage2)
        self._lbl_stage2 = QLabel("Run Stage 1 and review vacuoles first.")
        self._lbl_stage2.setWordWrap(True)
        s2_layout.addWidget(self._lbl_stage2)
        self._btn_review_para = QPushButton("🔍 Review parasites")
        self._btn_review_para.clicked.connect(self._open_parasite_curation)
        self._btn_review_para.setEnabled(False)
        s2_layout.addWidget(self._btn_review_para)
        layout.addWidget(s2_box)

        # Export
        exp_box = QGroupBox("Export")
        exp_layout = QVBoxLayout(exp_box)
        self._btn_table = QPushButton("📊 Show measurements table")
        self._btn_table.clicked.connect(self._show_table)
        self._btn_table.setEnabled(False)
        self._btn_save_csv = QPushButton("💾 Export measurements CSV")
        self._btn_save_csv.clicked.connect(self._export_csv)
        self._btn_save_csv.setEnabled(False)
        exp_layout.addWidget(self._btn_table)
        exp_layout.addWidget(self._btn_save_csv)
        layout.addWidget(exp_box)

        # Classifier status
        clf_box = QGroupBox("Classifier status")
        clf_layout = QVBoxLayout(clf_box)
        self._clf_status = QLabel("No classifier trained yet.")
        self._clf_status.setWordWrap(True)
        clf_layout.addWidget(self._clf_status)
        layout.addWidget(clf_box)

        # Log
        log_box = QGroupBox("Log")
        log_layout = QVBoxLayout(log_box)
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMinimumHeight(100)
        log_layout.addWidget(self._log)
        layout.addWidget(log_box)

        layout.addStretch()
        return w

    # ── Training tab ─────────────────────────────────────────────────────────

    def _build_training_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        # Vacuole model
        vac_box = QGroupBox("Vacuole model")
        vac_layout = QVBoxLayout(vac_box)
        self._sd_vac_count = QLabel("Training pairs: —")
        vac_layout.addWidget(self._sd_vac_count)
        vac_btn_row = QHBoxLayout()
        self._btn_refresh_vac = QPushButton("↺ Refresh")
        self._btn_refresh_vac.clicked.connect(
            lambda: self._refresh_training_count("vacuoles")
        )
        self._btn_clear_vac = QPushButton("🗑 Clear pairs")
        self._btn_clear_vac.clicked.connect(
            lambda: self._clear_training("vacuoles", clear_model=False)
        )
        self._btn_clear_vac_model = QPushButton("🗑 + model")
        self._btn_clear_vac_model.clicked.connect(
            lambda: self._clear_training("vacuoles", clear_model=True)
        )
        vac_btn_row.addWidget(self._btn_refresh_vac)
        vac_btn_row.addWidget(self._btn_clear_vac)
        vac_btn_row.addWidget(self._btn_clear_vac_model)
        vac_layout.addLayout(vac_btn_row)
        layout.addWidget(vac_box)

        # Parasite model
        para_box = QGroupBox("Parasite model")
        para_layout = QVBoxLayout(para_box)
        self._sd_para_count = QLabel("Training pairs: —")
        para_layout.addWidget(self._sd_para_count)
        para_btn_row = QHBoxLayout()
        self._btn_refresh_para = QPushButton("↺ Refresh")
        self._btn_refresh_para.clicked.connect(
            lambda: self._refresh_training_count("parasites")
        )
        self._btn_clear_para = QPushButton("🗑 Clear pairs")
        self._btn_clear_para.clicked.connect(
            lambda: self._clear_training("parasites", clear_model=False)
        )
        self._btn_clear_para_model = QPushButton("🗑 + model")
        self._btn_clear_para_model.clicked.connect(
            lambda: self._clear_training("parasites", clear_model=True)
        )
        para_btn_row.addWidget(self._btn_refresh_para)
        para_btn_row.addWidget(self._btn_clear_para)
        para_btn_row.addWidget(self._btn_clear_para_model)
        para_layout.addLayout(para_btn_row)
        layout.addWidget(para_box)

        # Load previous results
        load_box = QGroupBox("Load previous results")
        load_layout = QVBoxLayout(load_box)
        load_layout.addWidget(
            QLabel("Re-open a previously processed image for curation or export.")
        )
        stem_row = QHBoxLayout()
        stem_row.addWidget(QLabel("Result:"))
        self._load_stem_combo = QComboBox()
        self._load_stem_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        stem_row.addWidget(self._load_stem_combo, stretch=1)
        self._btn_refresh_stems = QPushButton("↺")
        self._btn_refresh_stems.setFixedWidth(26)
        self._btn_refresh_stems.clicked.connect(self._refresh_stems)
        stem_row.addWidget(self._btn_refresh_stems)
        load_layout.addLayout(stem_row)
        self._btn_load_results = QPushButton("📂 Load & open curation")
        self._btn_load_results.clicked.connect(self._load_previous_results)
        load_layout.addWidget(self._btn_load_results)
        layout.addWidget(load_box)

        layout.addStretch()

        # Initial refresh
        self._refresh_training_count("vacuoles")
        self._refresh_training_count("parasites")
        self._refresh_stems()

        return w

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _log_msg(self, msg: str) -> None:
        self._log.append(msg)

    def _populate_layers(self) -> None:
        from napari.layers import Image

        current = self._layer_combo.currentText()
        self._layer_combo.clear()
        for layer in self._viewer.layers:
            if isinstance(layer, Image):
                self._layer_combo.addItem(layer.name)
        idx = self._layer_combo.findText(current)
        if idx >= 0:
            self._layer_combo.setCurrentIndex(idx)
        self._autofill_pixel_size()

    def _autofill_pixel_size(self) -> None:
        if self._pixel_size.value() > 0:
            return
        from napari.layers import Image

        name = self._layer_combo.currentText()
        if not name or name not in self._viewer.layers:
            return
        layer = self._viewer.layers[name]
        if not isinstance(layer, Image):
            return
        scale = layer.scale
        px = float(scale[-1]) if len(scale) >= 1 else 0.0
        if 0.01 < px < 100.0 and not all(s == 1.0 for s in scale):
            self._pixel_size.setValue(round(px, 4))
            self._log_msg(f"Pixel size auto-read: {px:.4f} µm/px")

    def _get_image_array(self) -> np.ndarray:
        name = self._layer_combo.currentText()
        if not name:
            raise RuntimeError("No image layer selected.")
        layer = self._viewer.layers[name]
        data = np.asarray(layer.data).astype(np.float32)
        if data.ndim == 2:
            return data[..., np.newaxis]
        elif data.ndim == 3:
            if data.shape[0] < data.shape[1] and data.shape[0] < data.shape[2]:
                return np.moveaxis(data, 0, -1)
            return data
        elif data.ndim == 4:
            mid = data.shape[0] // 2
            plane = data[mid]
            if plane.shape[0] < plane.shape[1]:
                return np.moveaxis(plane, 0, -1)
            return plane.astype(np.float32)
        else:
            raise RuntimeError(f"Unsupported image shape: {data.shape}")

    def _browse_annot_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select annotations directory")
        if d:
            self._annot_dir.setText(d)

    def _pixel_area_limits(self) -> tuple[float, float]:
        px = self._pixel_size.value()
        if px > 0:
            return (
                self._min_area_um2.value() / (px**2),
                self._max_area_um2.value() / (px**2),
            )
        return 0.0, 1e9

    def _try_load_classifier(self) -> None:
        from ._learning import load_classifier

        clf = load_classifier(self._annot_dir.text())
        self._classifier = clf
        self._update_clf_status()

    def _update_clf_status(self) -> None:
        from ._learning import classifier_stats

        annot_dir = self._annot_dir.text()
        csv_path = Path(annot_dir) / "curated_features.csv"
        stats = classifier_stats(csv_path)
        if self._classifier is not None:
            self._clf_status.setText(
                f"Classifier loaded — {stats['total']} examples "
                f"({stats['accepted']} accepted, {stats['rejected']} rejected)."
            )
        elif stats["total"] > 0:
            self._clf_status.setText(
                f"{stats['total']} annotations on disk but no classifier yet."
            )
        else:
            self._clf_status.setText("No classifier trained yet.")

    # ── Stage 1: vacuole detection ───────────────────────────────────────────

    def _run_stage1(self) -> None:
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
                "Cancel or enter 0 to disable area filter:",
                decimals=4,
                min=0.0,
                max=100.0,
                value=0.105,
            )
            if ok and val > 0:
                self._pixel_size.setValue(val)

        min_area_px, max_area_px = self._pixel_area_limits()
        seg_backend = self._seg_backend.currentIndex()
        diameter = self._diameter.value()

        params = {
            "image": image,
            "seg_ch": self._seg_ch.value(),
            "seg_backend": seg_backend,
            "annot_dir": self._annot_dir.text(),
            "min_area_px": min_area_px,
            "max_area_px": max_area_px,
            "max_eccentricity": self._max_eccentricity.value(),
            "min_solidity": self._min_solidity.value(),
            "use_composite": self._use_composite.isChecked(),
            "diameter": diameter if diameter > 0 else None,
            "flow_threshold": self._flow_thresh.value(),
            "cellprob_threshold": self._cellprob_thresh.value(),
            "threshold_method": self._thresh_method.currentText(),
            "threshold_channel": self._thresh_channel.value(),
            "threshold_value": self._thresh_value.value(),
            "threshold_percentile": self._thresh_percentile.value(),
            "watershed_split": self._watershed_split.isChecked(),
            "watershed_min_distance": self._watershed_min_dist.value(),
        }

        self._btn_stage1.setEnabled(False)
        self._btn_stage1.setText("Running…")
        self._lbl_stage1.setText("Detecting vacuoles…")

        if seg_backend == 0:
            from ._segment import preload_model

            self._log_msg(preload_model())

        self._thread = QThread()
        self._worker = _VacuoleWorker(params)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_stage1_done)
        self._worker.error.connect(self._on_worker_error)
        self._worker.progress.connect(self._log_msg)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.start()

    def _on_stage1_done(
        self, raw_vac_labels: np.ndarray, vac_labels: np.ndarray
    ) -> None:
        self._vac_labels = vac_labels
        stem = self._image_stem

        raw_name = f"{stem}_vac_candidates"
        if raw_name in self._viewer.layers:
            self._viewer.layers[raw_name].data = raw_vac_labels
        else:
            self._viewer.add_labels(raw_vac_labels, name=raw_name, opacity=0.2)

        vac_name = f"{stem}_vacuoles"
        if vac_name in self._viewer.layers:
            self._viewer.layers[vac_name].data = vac_labels
        else:
            self._viewer.add_labels(vac_labels, name=vac_name)

        n = int(vac_labels.max())
        self._lbl_stage1.setText(f"Found {n} vacuoles.")
        self._log_msg(f"Stage 1 done — {n} vacuoles.")
        self._btn_stage1.setEnabled(True)
        self._btn_stage1.setText("▶ Detect vacuoles")
        self._btn_review_vac.setEnabled(True)

    def _on_worker_error(self, msg: str) -> None:
        self._log_msg(f"Error:\n{msg}")
        self._btn_stage1.setEnabled(True)
        self._btn_stage1.setText("▶ Detect vacuoles")
        self._btn_stage2.setEnabled(self._vac_labels is not None)

    # ── Vacuole curation ─────────────────────────────────────────────────────

    def _open_vacuole_curation(self) -> None:
        if self._vac_labels is None:
            self._log_msg("Run Stage 1 first.")
            return
        try:
            image = self._get_image_array()
        except RuntimeError:
            image = np.zeros((*self._vac_labels.shape, 2), dtype=np.float32)

        from ._curation import VacuoleCurationWidget

        vac_layer_name = f"{self._image_stem}_vacuoles"
        self._vac_curation_win = VacuoleCurationWidget(
            vac_labels=self._vac_labels,
            image=image,
            ch_cptsa=self._ch_cptsa.value(),
            ch_mcherry=self._ch_mcherry.value(),
            on_save=self._on_vacuole_curation_saved,
            viewer=self._viewer,
            labels_layer_name=vac_layer_name,
            parent=None,
        )
        self._vac_curation_win.setWindowTitle("Peredox — Vacuole Review (Stage 1)")
        self._vac_curation_win.resize(360, 560)
        self._vac_curation_win.show()

    def _on_vacuole_curation_saved(
        self, decisions: dict, curated_vac_labels: np.ndarray
    ) -> None:
        """Called when user saves vacuole curation."""
        # Update internal vacuole labels with any polygon edits
        self._vac_labels = curated_vac_labels

        # Save training pair for vacuole model
        try:
            image = self._get_image_array()
            annot_dir = self._annot_dir.text()
            from ._io import save_training_pair
            from ._stardist import count_training_pairs

            training_dir = Path(annot_dir) / "training_data"
            save_training_pair(
                image=image,
                labels=curated_vac_labels,
                decisions=decisions,
                stem=self._image_stem,
                training_dir=training_dir,
                vacuole_map={
                    int(v): int(v) for v in np.unique(curated_vac_labels) if v != 0
                },
            )
            n_vac = count_training_pairs(training_dir, mode="vacuoles")
            self._log_msg(f"Vacuole training pair saved ({n_vac} total).")
            self._refresh_training_count("vacuoles")
        except Exception as exc:
            self._log_msg(f"Training pair save error: {exc}")

        # Remove rejected vacuoles from the label image
        rejected = {vid for vid, dec in decisions.items() if dec == 0}
        if rejected:
            for vid in rejected:
                self._vac_labels[self._vac_labels == vid] = 0

        n_accepted = int(len([d for d in decisions.values() if d == 1]))
        self._log_msg(f"Vacuole curation saved — {n_accepted} accepted.")
        self._lbl_stage1.setText(
            f"Curation done — {n_accepted} accepted vacuoles ready for Stage 2."
        )
        self._btn_stage2.setEnabled(True)
        self._lbl_stage2.setText("Ready — click to detect parasites.")

    # ── Stage 2: parasite detection ──────────────────────────────────────────

    def _run_stage2(self) -> None:
        if self._vac_labels is None:
            self._log_msg("Complete Stage 1 and vacuole review first.")
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

        min_area_px, max_area_px = self._pixel_area_limits()
        px = self._pixel_size.value()

        params = {
            "image": image,
            "vac_labels": self._vac_labels,
            "seg_ch": self._seg_ch.value(),
            "seg_backend": self._seg_backend.currentIndex(),
            "annot_dir": self._annot_dir.text(),
            "ch_cptsa": self._ch_cptsa.value(),
            "ch_mcherry": self._ch_mcherry.value(),
            "ch_names": ch_names,
            "pixel_size": px,
            "min_area_px": min_area_px,
            "max_area_px": max_area_px,
            "max_eccentricity": self._max_eccentricity.value(),
            "min_solidity": self._min_solidity.value(),
            "vacuole_method": self._vacuole_method.currentText(),
        }

        self._btn_stage2.setEnabled(False)
        self._btn_stage2.setText("Running…")
        self._lbl_stage2.setText("Detecting parasites…")

        self._thread = QThread()
        self._worker = _ParasiteWorker(params)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_stage2_done)
        self._worker.error.connect(self._on_worker_error)
        self._worker.progress.connect(self._log_msg)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.start()

    def _on_stage2_done(
        self,
        para_labels: np.ndarray,
        vacuole_map: dict,
        features,
        measurements,
    ) -> None:
        self._para_labels = para_labels
        self._vac_map = vacuole_map
        self._features = features
        self._measurements = measurements
        stem = self._image_stem

        para_name = f"{stem}_parasites"
        if para_name in self._viewer.layers:
            self._viewer.layers[para_name].data = para_labels
        else:
            self._viewer.add_labels(para_labels, name=para_name)

        n_para = int(para_labels.max())
        n_vac = len(set(vacuole_map.values()))
        self._lbl_stage2.setText(f"Found {n_para} parasites in {n_vac} vacuoles.")
        self._btn_stage2.setEnabled(True)
        self._btn_stage2.setText("▶ Detect parasites in vacuoles")
        self._btn_review_para.setEnabled(True)
        self._btn_table.setEnabled(True)
        self._btn_save_csv.setEnabled(True)

    # ── Parasite curation ────────────────────────────────────────────────────

    def _open_parasite_curation(self) -> None:
        if self._para_labels is None:
            self._log_msg("Run Stage 2 first.")
            return
        try:
            image = self._get_image_array()
        except RuntimeError:
            image = np.zeros((*self._para_labels.shape, 2), dtype=np.float32)

        n_ch = image.shape[-1]
        ch_names = {i: f"ch{i}" for i in range(n_ch)}
        ch_names[self._ch_cptsa.value()] = "cptsa"
        ch_names[self._ch_mcherry.value()] = "mcherry"
        px = self._pixel_size.value()

        from ._curation import CurationWidget

        para_layer_name = f"{self._image_stem}_parasites"
        self._para_curation_win = CurationWidget(
            labels=self._para_labels,
            image=image,
            measurements=self._measurements,
            ch_cptsa=self._ch_cptsa.value(),
            ch_mcherry=self._ch_mcherry.value(),
            ch_names=ch_names,
            pixel_size_um=px if px > 0 else None,
            vacuole_assignments=self._vac_map if self._vac_map else None,
            on_save=self._on_parasite_curation_saved,
            viewer=self._viewer,
            labels_layer_name=para_layer_name,
            parent=None,
        )
        self._para_curation_win.setWindowTitle("Peredox — Parasite Review (Stage 2)")
        self._para_curation_win.resize(380, 620)
        self._para_curation_win.show()

    def _on_parasite_curation_saved(
        self, decisions: dict, vacuole_assignments: dict | None = None
    ) -> None:
        from ._io import append_curated_annotations, save_training_pair
        from ._learning import train_classifier
        from ._stardist import count_training_pairs

        annot_dir = self._annot_dir.text()
        csv_path = append_curated_annotations(
            decisions=decisions,
            features=self._features if self._features is not None else {},
            image_stem=self._image_stem,
            annotations_dir=annot_dir,
            vacuole_assignments=vacuole_assignments,
        )
        n_dec = sum(1 for v in decisions.values() if v in (0, 1))
        self._log_msg(f"Saved {n_dec} annotations → {csv_path}")

        # Save both training pairs
        if self._para_labels is not None:
            try:
                image = self._get_image_array()
                training_dir = Path(annot_dir) / "training_data"
                save_training_pair(
                    image=image,
                    labels=self._para_labels,
                    decisions=decisions,
                    stem=self._image_stem,
                    training_dir=training_dir,
                    vacuole_map=self._vac_map if self._vac_map else None,
                )
                n_vac = count_training_pairs(training_dir, mode="vacuoles")
                n_para = count_training_pairs(training_dir, mode="parasites")
                self._log_msg(
                    f"Training pairs saved — vacuoles: {n_vac}, parasites: {n_para}."
                )
                self._refresh_training_count("vacuoles")
                self._refresh_training_count("parasites")
            except Exception as exc:
                self._log_msg(f"Training pair save error: {exc}")

        # Retrain RandomForest classifier
        clf = train_classifier(csv_path)
        if clf is not None:
            self._classifier = clf
            self._log_msg("Classifier retrained.")
        else:
            self._log_msg("Not enough data to retrain classifier yet.")
        self._update_clf_status()

    # ── Measurements & export ────────────────────────────────────────────────

    def _show_table(self) -> None:
        if self._measurements is None or len(self._measurements) == 0:
            self._log_msg("No measurements available.")
            return
        df = self._measurements.reset_index()
        win = QWidget(self, Qt.Window)
        win.setWindowTitle("PV Measurements")
        win.resize(900, 400)
        layout = QVBoxLayout(win)
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
        win.show()

    def _export_csv(self) -> None:
        from ._io import save_labels, save_measurements

        if self._measurements is None:
            return
        annot_dir = self._annot_dir.text()
        meas_path = save_measurements(self._measurements, self._image_stem, annot_dir)
        if self._para_labels is not None:
            save_labels(self._para_labels, self._image_stem, annot_dir)
        self._log_msg(f"Saved → {meas_path}")

    # ── Training management ───────────────────────────────────────────────────

    def _refresh_training_count(self, mode: str) -> None:
        from ._stardist import count_training_pairs

        annot_dir = self._annot_dir.text()
        training_dir = Path(annot_dir) / "training_data"
        n = count_training_pairs(training_dir, mode=mode)
        if mode == "vacuoles":
            self._sd_vac_count.setText(f"Training pairs: {n}")
        else:
            self._sd_para_count.setText(f"Training pairs: {n}")

    def _clear_training(self, mode: str, clear_model: bool) -> None:
        from ._io import clear_training_data

        annot_dir = self._annot_dir.text()
        msg = f"Delete all '{mode}' training pairs" + (
            " AND the trained model?" if clear_model else "?"
        )
        reply = QMessageBox.question(
            self,
            "Clear training data?",
            msg + "\n\nThis cannot be undone.",
            QMessageBox.Yes | QMessageBox.Cancel,
        )
        if reply == QMessageBox.Yes:
            n = clear_training_data(annot_dir, mode=mode, clear_model=clear_model)
            self._log_msg(f"Cleared {n} TIFF files ({mode}).")
            self._refresh_training_count(mode)

    # ── Load previous results ─────────────────────────────────────────────────

    def _refresh_stems(self) -> None:
        annot_dir = self._annot_dir.text()
        results_dir = Path(annot_dir) / "results"
        self._load_stem_combo.clear()
        if not results_dir.exists():
            return
        stems = sorted(
            {
                p.stem.replace("_measurements", "")
                for p in results_dir.glob("*_measurements.csv")
            }
        )
        for s in stems:
            self._load_stem_combo.addItem(s)

    def _load_previous_results(self) -> None:
        from ._io import load_labels, load_measurements
        from ._segment import group_by_vacuole

        stem = self._load_stem_combo.currentText()
        if not stem:
            self._log_msg("No result selected.")
            return
        annot_dir = self._annot_dir.text()
        labels = load_labels(stem, annot_dir)
        measurements = load_measurements(stem, annot_dir)
        if labels is None:
            self._log_msg(f"Labels TIFF not found for '{stem}'.")
            return
        if measurements is None:
            self._log_msg(f"Measurements CSV not found for '{stem}'.")
            return

        self._para_labels = labels
        self._measurements = measurements
        self._image_stem = stem
        self._features = None
        self._vac_map = group_by_vacuole(labels, dilation_px=5)

        layer_name = f"{stem}_parasites"
        if layer_name in self._viewer.layers:
            self._viewer.layers[layer_name].data = labels
        else:
            self._viewer.add_labels(labels, name=layer_name)

        self._log_msg(
            f"Loaded '{stem}': {int(labels.max())} labels, {len(measurements)} rows."
        )
        self._btn_table.setEnabled(True)
        self._btn_save_csv.setEnabled(True)
        self._btn_review_para.setEnabled(True)
        self._open_parasite_curation()

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
        self._btn_host_stage2.setEnabled(False)
        self._btn_host_review_para.setEnabled(False)
        self._btn_host_measure.setEnabled(False)
        self._btn_host_export.setEnabled(False)
        self._lbl_host_stage1.setText("Segmenting host cells…")

        from ._segment import preload_model

        self._log_msg(preload_model())

        self._host_thread = QThread()
        self._host_worker = _HostWorker(params)
        self._host_worker.moveToThread(self._host_thread)
        self._host_thread.started.connect(self._host_worker.run)
        self._host_worker.finished.connect(self._on_host_stage1_done)
        self._host_worker.error.connect(self._on_host_worker_error)
        self._host_worker.progress.connect(self._log_msg)
        self._host_worker.finished.connect(self._host_thread.quit)
        self._host_worker.error.connect(self._host_thread.quit)
        self._host_thread.start()

    def _on_host_stage1_done(
        self, raw_host_labels: np.ndarray, host_labels: np.ndarray
    ) -> None:
        self._host_labels = host_labels
        # Reset downstream host state — new hosts invalidate old parasites
        self._host_para_labels = None
        self._host_vac_map = {}
        self._host_measurements = None
        self._host_review_saved = False
        self._host_para_measurements = None
        self._host_features = None
        self._btn_host_stage2.setEnabled(False)
        self._btn_host_review_para.setEnabled(False)
        self._btn_host_measure.setEnabled(False)
        self._btn_host_export.setEnabled(False)
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

    def _on_host_worker_error(self, msg: str) -> None:
        self._log_msg(f"Error:\n{msg}")
        self._btn_host_stage1.setEnabled(True)
        self._btn_host_stage1.setText("▶ Segment host cells")
        self._btn_host_stage2.setEnabled(
            self._host_labels is not None and self._host_review_saved
        )
        self._btn_host_stage2.setText("▶ Segment parasites in hosts")

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
        self._host_review_saved = True
        self._btn_host_stage2.setEnabled(n_remaining > 0)
        if n_remaining > 0:
            self._lbl_host_stage2.setText("Ready — click to detect parasites.")
        # Invalidate any parasites detected against the pre-curation hosts
        self._host_para_labels = None
        self._btn_host_review_para.setEnabled(False)
        self._btn_host_measure.setEnabled(False)
        self._host_measurements = None
        self._host_para_measurements = None
        self._btn_host_export.setEnabled(False)

    # ── Placeholders completed in Tasks 10–11 ────────────────────────────────

    def _run_host_stage2(self) -> None:
        if self._host_labels is None or self._host_labels.max() == 0:
            self._log_msg("Segment and review host cells first.")
            return
        if not self._host_review_saved:
            self._log_msg(
                "Save the host review first (Stage H2 runs on accepted hosts only)."
            )
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
        self._btn_host_stage1.setEnabled(False)
        self._lbl_host_stage2.setText("Detecting parasites in hosts…")

        if self._seg_backend.currentIndex() == 0:
            from ._segment import preload_model

            self._log_msg(preload_model())

        self._host_thread = QThread()
        self._host_worker = _HostParasiteWorker(params)
        self._host_worker.moveToThread(self._host_thread)
        self._host_thread.started.connect(self._host_worker.run)
        self._host_worker.finished.connect(self._on_host_stage2_done)
        self._host_worker.error.connect(self._on_host_worker_error)
        self._host_worker.progress.connect(self._log_msg)
        self._host_worker.finished.connect(self._host_thread.quit)
        self._host_worker.error.connect(self._host_thread.quit)
        self._host_thread.start()

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
        self._btn_host_stage1.setEnabled(True)
        self._btn_host_review_para.setEnabled(True)
        self._btn_host_measure.setEnabled(True)
        self._lbl_host_measure.setText("Ready — assign parasites and measure hosts.")

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
            # Rejected / background-dropped parasites have no host_id (NaN) —
            # exclude them, matching the batch path's para_to_host semantics.
            self._host_para_measurements = self._host_para_measurements[
                self._host_para_measurements["host_id"].notna()
            ]

        n_inf = int(hosts_df["infected"].sum()) if not hosts_df.empty else 0
        n_tot = len(hosts_df)
        n_empty = int(hosts_df["cytosol_empty"].sum()) if not hosts_df.empty else 0
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


# ---------------------------------------------------------------------------
# napari entry point
# ---------------------------------------------------------------------------


def make_main_widget(napari_viewer: napari.Viewer) -> PeredoxWidget:
    """Called by napari when the user opens 'Peredox: Segment & Measure PVs'."""
    return PeredoxWidget(napari_viewer)
