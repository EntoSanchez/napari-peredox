"""
PV curation widget — lets the user page through each detected PV and
accept or reject it.

UI layout
---------
  [← Prev]  PV 3 / 12   [Next →]
  ┌────────────────────────────┐
  │   thumbnail (cpTSa + merge)│
  └────────────────────────────┘
  Area: 1 234 px   Ratio: 1.42
  [✓ Accept]   [✗ Reject]   [Skip]
  ──────────────────────────────
  Status: 7 accepted, 3 rejected, 2 pending
  [Save & retrain classifier]

Accepted/rejected decisions are written back to the main widget's
`_curation_state` dict so that _io.py can persist them.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QImage, QPixmap
from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    import pandas as pd


def _array_to_pixmap(rgb: np.ndarray, size: int = 220) -> QPixmap:
    """Convert an (H, W, 3) uint8 array to a scaled QPixmap."""
    h, w = rgb.shape[:2]
    qimg = QImage(rgb.tobytes(), w, h, w * 3, QImage.Format_RGB888)
    pix = QPixmap.fromImage(qimg)
    return pix.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)


def _make_thumbnail(
    image: np.ndarray,
    labels: np.ndarray,
    label_id: int,
    ch_cptsa: int,
    ch_mcherry: int,
    pad: int = 20,
) -> np.ndarray:
    """
    Crop a padded bounding box around label_id and return an RGB (H, W, 3) uint8.
    Green channel = cpTSapphire, Red channel = mCherry, Blue = 0.
    The mask boundary is drawn in white.
    """
    from skimage.segmentation import find_boundaries

    ys, xs = np.where(labels == label_id)
    if len(ys) == 0:
        return np.zeros((64, 64, 3), dtype=np.uint8)

    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, labels.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, labels.shape[1])

    if image.ndim == 2:
        image = image[..., np.newaxis]

    n_ch = image.shape[-1]

    def _norm_crop(ch):
        if ch >= n_ch:
            return np.zeros((y1 - y0, x1 - x0), dtype=np.float32)
        crop = image[y0:y1, x0:x1, ch].astype(np.float32)
        mn, mx = crop.min(), crop.max()
        return (crop - mn) / (mx - mn + 1e-9)

    green = _norm_crop(ch_cptsa)
    red = _norm_crop(ch_mcherry)
    blue = np.zeros_like(green)

    rgb = np.stack([red, green, blue], axis=-1)

    # Overlay mask boundary in white
    mask_crop = labels[y0:y1, x0:x1] == label_id
    boundary = find_boundaries(mask_crop, mode="outer")
    rgb[boundary] = [1.0, 1.0, 1.0]

    return (rgb * 255).clip(0, 255).astype(np.uint8)


class CurationWidget(QWidget):
    """
    Dock widget for accepting/rejecting individual PV segments.

    Signals
    -------
    curation_saved : emitted when the user clicks 'Save & retrain'.
    """

    curation_saved = Signal()

    def __init__(
        self,
        labels: np.ndarray,
        image: np.ndarray,
        measurements: pd.DataFrame,
        ch_cptsa: int,
        ch_mcherry: int,
        on_save: Callable | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self._labels = labels
        self._image = image
        self._measurements = measurements
        self._ch_cptsa = ch_cptsa
        self._ch_mcherry = ch_mcherry
        self._on_save = on_save

        self._label_ids: list[int] = (
            measurements.index.tolist()
            if measurements is not None and len(measurements) > 0
            else []
        )
        # decisions: 1=accept, 0=reject, -1=pending
        self._decisions: dict[int, int] = {lid: -1 for lid in self._label_ids}
        self._current_idx: int = 0

        # vacuole_assignments: maps each parasite label → vacuole ID.
        # Pre-populated from the 'vacuole_id' column in measurements if present;
        # otherwise each parasite is treated as its own vacuole.
        if measurements is not None and "vacuole_id" in measurements.columns:
            self._vacuole_assignments: dict[int, int] = {
                lid: int(measurements.loc[lid, "vacuole_id"])
                for lid in self._label_ids
                if lid in measurements.index
            }
        else:
            self._vacuole_assignments = {
                lid: i + 1 for i, lid in enumerate(self._label_ids)
            }

        self._build_ui()
        self._refresh()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Navigation row
        nav = QHBoxLayout()
        self._btn_prev = QPushButton("← Prev")
        self._btn_prev.clicked.connect(self._prev)
        self._lbl_nav = QLabel()
        self._lbl_nav.setAlignment(Qt.AlignCenter)
        self._btn_next = QPushButton("Next →")
        self._btn_next.clicked.connect(self._next)
        nav.addWidget(self._btn_prev)
        nav.addWidget(self._lbl_nav, stretch=1)
        nav.addWidget(self._btn_next)
        layout.addLayout(nav)

        # Thumbnail
        self._thumbnail_label = QLabel()
        self._thumbnail_label.setAlignment(Qt.AlignCenter)
        self._thumbnail_label.setFixedSize(240, 240)
        self._thumbnail_label.setStyleSheet("background-color: black;")
        layout.addWidget(self._thumbnail_label, alignment=Qt.AlignHCenter)

        # Measurement info
        self._info_label = QLabel()
        self._info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._info_label)

        # Vacuole ID — auto-assigned by proximity; user can override
        vac_row = QHBoxLayout()
        vac_row.addWidget(QLabel("Vacuole ID:"))
        self._vacuole_spin = QSpinBox()
        self._vacuole_spin.setRange(0, 9999)
        self._vacuole_spin.setToolTip(
            "Auto-assigned vacuole group ID.\n"
            "Parasites sharing the same ID belong to the same PV.\n"
            "Edit this to correct the grouping before saving."
        )
        self._vacuole_spin.valueChanged.connect(self._on_vacuole_id_changed)
        vac_row.addWidget(self._vacuole_spin)
        vac_row.addStretch()
        layout.addLayout(vac_row)

        # Decision buttons
        btn_row = QHBoxLayout()
        self._btn_accept = QPushButton("✓ Accept")
        self._btn_accept.setStyleSheet("background-color: #2e7d32; color: white;")
        self._btn_accept.clicked.connect(self._accept)
        self._btn_reject = QPushButton("✗ Reject")
        self._btn_reject.setStyleSheet("background-color: #c62828; color: white;")
        self._btn_reject.clicked.connect(self._reject)
        self._btn_skip = QPushButton("Skip")
        self._btn_skip.clicked.connect(self._next)
        btn_row.addWidget(self._btn_accept)
        btn_row.addWidget(self._btn_reject)
        btn_row.addWidget(self._btn_skip)
        layout.addLayout(btn_row)

        # Status
        self._status_label = QLabel()
        self._status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._status_label)

        # Save button
        self._btn_save = QPushButton("💾 Save & retrain classifier")
        self._btn_save.clicked.connect(self._save)
        layout.addWidget(self._btn_save)

        layout.addStretch()

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _prev(self):
        if self._current_idx > 0:
            self._current_idx -= 1
            self._refresh()

    def _next(self):
        if self._current_idx < len(self._label_ids) - 1:
            self._current_idx += 1
        self._refresh()

    def _accept(self):
        if self._label_ids:
            lid = self._label_ids[self._current_idx]
            self._decisions[lid] = 1
            # Advance if not on the last item; otherwise stay and refresh so the
            # status label and decision text update to show "✓ Accepted".
            if self._current_idx < len(self._label_ids) - 1:
                self._current_idx += 1
            self._refresh()

    def _reject(self):
        if self._label_ids:
            lid = self._label_ids[self._current_idx]
            self._decisions[lid] = 0
            if self._current_idx < len(self._label_ids) - 1:
                self._current_idx += 1
            self._refresh()

    def _on_vacuole_id_changed(self, value: int):
        """Record the user-edited vacuole ID for the currently displayed parasite."""
        if self._label_ids:
            lid = self._label_ids[self._current_idx]
            self._vacuole_assignments[lid] = value

    def _save(self):
        if self._on_save:
            self._on_save(self._decisions, self._vacuole_assignments)
        self.curation_saved.emit()

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def _refresh(self):
        n = len(self._label_ids)
        if n == 0:
            self._lbl_nav.setText("No segments")
            self._info_label.setText("")
            self._thumbnail_label.clear()
            self._vacuole_spin.setValue(0)
            self._update_status()
            return

        idx = self._current_idx
        lid = self._label_ids[idx]
        self._lbl_nav.setText(f"Parasite {idx + 1} / {n}  (id={lid})")

        # Thumbnail
        thumb = _make_thumbnail(
            self._image, self._labels, lid, self._ch_cptsa, self._ch_mcherry
        )
        self._thumbnail_label.setPixmap(_array_to_pixmap(thumb, 230))

        # Info
        row = (
            self._measurements.loc[lid]
            if self._measurements is not None and lid in self._measurements.index
            else None
        )
        if row is not None:
            area = int(row.get("area_px", 0))
            ratio = row.get("ratio_cptsa_mcherry", float("nan"))
            ratio_str = f"{ratio:.3f}" if not np.isnan(ratio) else "—"
            n_in_vac = int(row.get("parasites_per_vacuole", 1))
            decision = {1: "✓ Accepted", 0: "✗ Rejected", -1: "Pending"}[
                self._decisions[lid]
            ]
            self._info_label.setText(
                f"Area: {area:,} px   Ratio: {ratio_str}   In vacuole: {n_in_vac}\n{decision}"
            )
        else:
            self._info_label.setText("")

        # Sync vacuole ID spinbox (suppress valueChanged so we don't overwrite)
        self._vacuole_spin.blockSignals(True)
        self._vacuole_spin.setValue(self._vacuole_assignments.get(lid, lid))
        self._vacuole_spin.blockSignals(False)

        self._update_status()

    def _update_status(self):
        n_accept = sum(v == 1 for v in self._decisions.values())
        n_reject = sum(v == 0 for v in self._decisions.values())
        n_pend = sum(v == -1 for v in self._decisions.values())
        self._status_label.setText(
            f"✓ {n_accept} accepted   ✗ {n_reject} rejected   … {n_pend} pending"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def decisions(self) -> dict[int, int]:
        """Return {label_id: decision} where decision is 1, 0, or -1 (pending)."""
        return dict(self._decisions)


def make_curation_widget(napari_viewer=None):
    """
    Entry point called by napari.  Returns a placeholder widget; the real
    curation widget is launched from the main Peredox widget after segmentation.
    """
    w = QWidget()
    lyt = QVBoxLayout(w)
    lbl = QLabel(
        "Run 'Peredox: Segment & Measure PVs' first,\n"
        "then click 'Open Curation Panel' to review segments."
    )
    lbl.setAlignment(Qt.AlignCenter)
    lbl.setWordWrap(True)
    lyt.addWidget(lbl)
    return w
