"""
PV curation widget — lets the user page through each detected PV and
accept or reject it.

UI layout
---------
  [← Prev]  PV 3 / 12   [Next →]
  ┌────────────────────────────┐
  │   thumbnail (cpTSa + merge)│   ← click here to place split seeds
  └────────────────────────────┘
  Area: 1 234 px   Ratio: 1.42
  Vacuole ID: [5]
  [✓ Accept]   [✗ Reject]   [Skip]
  [✂ Split segment]
  ──────────────────────────────
  Status: 7 accepted, 3 rejected, 2 pending
  [Save & retrain classifier]

Accepted/rejected decisions are written back to the main widget's
`_curation_state` dict so that _io.py can persist them.

Split workflow
--------------
Click '✂ Split segment' to enter draw mode.  Click and drag a line
across the boundary between the two merged parasites — an orange stroke
appears in real time.  Click 'Apply split': the drawn line is removed
from the mask and connected-component labelling separates the two
pieces into independent segments.  Tiny fragments along the cut are
absorbed into the nearest large piece.  'Cancel' discards the stroke.
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

# Display size passed to _array_to_pixmap
_THUMB_DISPLAY = 230
# Fixed size of the QLabel containing the thumbnail
_LABEL_SIZE = 240


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
    sibling_label_ids: list[int] | None = None,
    pad: int = 20,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """
    Crop a bounding box around label_id (and any sibling parasites sharing
    the same vacuole) and return an RGB (H, W, 3) uint8 array plus the crop
    coordinates (y0, x0, y1, x1).

    Green channel = cpTSapphire, Red channel = mCherry, Blue = 0.
    Selected parasite boundary → white.
    Sibling parasite boundaries → cyan, so the whole vacuole context is visible.
    """
    from skimage.segmentation import find_boundaries

    ys, xs = np.where(labels == label_id)
    if len(ys) == 0:
        return np.zeros((64, 64, 3), dtype=np.uint8), (0, 0, 64, 64)

    # Expand the bounding box to include all sibling parasites in this vacuole
    all_ys, all_xs = ys, xs
    for sib_id in sibling_label_ids or []:
        sib_ys, sib_xs = np.where(labels == sib_id)
        if len(sib_ys):
            all_ys = np.concatenate([all_ys, sib_ys])
            all_xs = np.concatenate([all_xs, sib_xs])

    y0 = max(int(all_ys.min()) - pad, 0)
    y1 = min(int(all_ys.max()) + pad + 1, labels.shape[0])
    x0 = max(int(all_xs.min()) - pad, 0)
    x1 = min(int(all_xs.max()) + pad + 1, labels.shape[1])

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

    # Draw sibling outlines in cyan first, then selected outline in white on top
    for sib_id in sibling_label_ids or []:
        sib_mask = labels[y0:y1, x0:x1] == sib_id
        if sib_mask.any():
            rgb[find_boundaries(sib_mask, mode="outer")] = [0.0, 0.75, 1.0]

    mask_crop = labels[y0:y1, x0:x1] == label_id
    rgb[find_boundaries(mask_crop, mode="outer")] = [1.0, 1.0, 1.0]

    return (rgb * 255).clip(0, 255).astype(np.uint8), (y0, x0, y1, x1)


def _bresenham(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    """Integer pixel positions on the line from (r0, c0) to (r1, c1)."""
    pts: list[tuple[int, int]] = []
    dr, dc = abs(r1 - r0), abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc
    while True:
        pts.append((r0, c0))
        if r0 == r1 and c0 == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r0 += sr
        if e2 < dr:
            err += dr
            c0 += sc
    return pts


class _ClickableThumbnail(QLabel):
    """
    QLabel that emits click and drag positions for the split draw tool.
    In draw mode the user click-drags a line across a merged segment.
    """

    clicked_at = Signal(int, int)
    dragged_to = Signal(int, int)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked_at.emit(event.x(), event.y())
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            self.dragged_to.emit(event.x(), event.y())
        super().mouseMoveEvent(event)


class CurationWidget(QWidget):
    """
    Dock widget for accepting/rejecting individual PV segments, with an
    interactive split tool to separate merged segments by clicking seeds.

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
        ch_names: dict[int, str] | None = None,
        pixel_size_um: float | None = None,
        vacuole_assignments: dict[int, int] | None = None,
        on_save: Callable | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self._labels = labels
        self._image = image
        self._measurements = measurements
        self._ch_cptsa = ch_cptsa
        self._ch_mcherry = ch_mcherry
        self._ch_names = ch_names or {ch_cptsa: "cptsa", ch_mcherry: "mcherry"}
        self._pixel_size_um = pixel_size_um
        self._on_save = on_save

        self._label_ids: list[int] = (
            measurements.index.tolist()
            if measurements is not None and len(measurements) > 0
            else []
        )
        self._decisions: dict[int, int] = {lid: -1 for lid in self._label_ids}
        self._current_idx: int = 0

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

        # If the caller passes the full label→vacuole map (e.g. from batch mode
        # where select_one_per_vacuole has reduced measurements to representatives
        # only), use it so that all siblings are visible in the thumbnail.
        if vacuole_assignments is not None:
            self._vacuole_assignments = dict(vacuole_assignments)

        # ── Split-mode state ──────────────────────────────────────────────────
        self._split_mode: bool = False
        # Freehand stroke drawn by the user (thumbnail QLabel coords)
        self._stroke_points: list[tuple[int, int]] = []
        # Crop coords for the currently displayed thumbnail
        self._thumb_crop: tuple[int, int, int, int] | None = None
        # Raw RGB thumbnail (before stroke overlay)
        self._thumb_raw: np.ndarray | None = None

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

        # Thumbnail — clickable so split-mode seeds can be placed
        self._thumbnail_label = _ClickableThumbnail()
        self._thumbnail_label.setAlignment(Qt.AlignCenter)
        self._thumbnail_label.setFixedSize(_LABEL_SIZE, _LABEL_SIZE)
        self._thumbnail_label.setStyleSheet("background-color: black;")
        self._thumbnail_label.clicked_at.connect(self._on_thumbnail_clicked)
        self._thumbnail_label.dragged_to.connect(self._on_stroke_drag)
        layout.addWidget(self._thumbnail_label, alignment=Qt.AlignHCenter)

        # Measurement info
        self._info_label = QLabel()
        self._info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._info_label)

        # Vacuole ID
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

        # ── Split section ─────────────────────────────────────────────────────
        self._btn_split_toggle = QPushButton("✂  Split segment")
        self._btn_split_toggle.setToolTip(
            "Enter draw mode: click and drag a line across the merged\n"
            "parasites to cut them apart, then click 'Apply split'."
        )
        self._btn_split_toggle.clicked.connect(self._toggle_split_mode)
        layout.addWidget(self._btn_split_toggle)

        # Instruction + apply/cancel row — hidden until split mode is active
        self._split_panel = QWidget()
        split_layout = QVBoxLayout(self._split_panel)
        split_layout.setContentsMargins(0, 0, 0, 0)
        split_layout.setSpacing(4)

        self._split_info = QLabel(
            "Click and drag across the two parasites to draw the cut line."
        )
        self._split_info.setAlignment(Qt.AlignCenter)
        self._split_info.setStyleSheet("color: #e6ac00; font-style: italic;")
        self._split_info.setWordWrap(True)
        split_layout.addWidget(self._split_info)

        split_btn_row = QHBoxLayout()
        self._btn_apply_split = QPushButton("Apply split")
        self._btn_apply_split.setStyleSheet("font-weight: bold;")
        self._btn_apply_split.clicked.connect(self._apply_split)
        self._btn_cancel_split = QPushButton("Cancel")
        self._btn_cancel_split.clicked.connect(self._cancel_split)
        split_btn_row.addWidget(self._btn_apply_split)
        split_btn_row.addWidget(self._btn_cancel_split)
        split_layout.addLayout(split_btn_row)

        self._split_panel.setVisible(False)
        layout.addWidget(self._split_panel)

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
            self._cancel_split()
            self._current_idx -= 1
            self._refresh()

    def _next(self):
        self._cancel_split()
        if self._current_idx < len(self._label_ids) - 1:
            self._current_idx += 1
        self._refresh()

    def _accept(self):
        if self._label_ids:
            self._cancel_split()
            lid = self._label_ids[self._current_idx]
            self._decisions[lid] = 1
            if self._current_idx < len(self._label_ids) - 1:
                self._current_idx += 1
            self._refresh()

    def _reject(self):
        if self._label_ids:
            self._cancel_split()
            lid = self._label_ids[self._current_idx]
            self._decisions[lid] = 0
            if self._current_idx < len(self._label_ids) - 1:
                self._current_idx += 1
            self._refresh()

    def _on_vacuole_id_changed(self, value: int):
        if self._label_ids:
            lid = self._label_ids[self._current_idx]
            self._vacuole_assignments[lid] = value

    def _save(self):
        if self._on_save:
            self._on_save(self._decisions, self._vacuole_assignments)
        self.curation_saved.emit()

    # ------------------------------------------------------------------
    # Split mode
    # ------------------------------------------------------------------

    def _toggle_split_mode(self):
        if self._split_mode:
            self._cancel_split()
        else:
            self._split_mode = True
            self._stroke_points = []
            self._split_panel.setVisible(True)
            self._btn_split_toggle.setText("✂  (draw mode ON — drag across segment)")
            self._btn_split_toggle.setStyleSheet("color: #e6ac00; font-weight: bold;")
            self._thumbnail_label.setCursor(Qt.CrossCursor)
            self._split_info.setText(
                "Click and drag across the two parasites to draw the cut line."
            )

    def _cancel_split(self):
        if not self._split_mode:
            return
        self._split_mode = False
        self._stroke_points = []
        self._split_panel.setVisible(False)
        self._btn_split_toggle.setText("✂  Split segment")
        self._btn_split_toggle.setStyleSheet("")
        self._thumbnail_label.setCursor(Qt.ArrowCursor)
        if self._thumb_raw is not None:
            self._thumbnail_label.setPixmap(
                _array_to_pixmap(self._thumb_raw, _THUMB_DISPLAY)
            )

    def _on_thumbnail_clicked(self, wx: int, wy: int):
        """Start a new draw stroke (resets any previous stroke)."""
        if not self._split_mode or self._thumb_crop is None or not self._label_ids:
            return
        self._stroke_points = [(wx, wy)]
        self._split_info.setText("Drag to draw the cut line…")
        self._redraw_thumbnail()

    def _on_stroke_drag(self, wx: int, wy: int):
        """Extend the current stroke while the mouse button is held."""
        if not self._split_mode or not self._stroke_points:
            return
        self._stroke_points.append((wx, wy))
        self._redraw_thumbnail()
        n = len(self._stroke_points)
        self._split_info.setText(
            f"Drawing… {n} points. Release mouse, then click 'Apply split'."
        )

    def _redraw_thumbnail(self):
        """Redraw thumbnail with the current cut stroke overlaid in orange."""
        if self._thumb_raw is None or self._thumb_crop is None:
            return

        rgb = self._thumb_raw.copy()
        half = 2  # stroke half-width in thumbnail pixels (5 px total)
        for wx, wy in self._stroke_points:
            for dr in range(-half, half + 1):
                for dc in range(-half, half + 1):
                    nr, nc = wy + dr, wx + dc
                    if 0 <= nr < rgb.shape[0] and 0 <= nc < rgb.shape[1]:
                        rgb[nr, nc] = [255, 140, 0]  # orange

        self._thumbnail_label.setPixmap(_array_to_pixmap(rgb, _THUMB_DISPLAY))

    def _apply_split(self):
        """Cut the segment along the drawn stroke using connected components."""
        if len(self._stroke_points) < 2:
            self._split_info.setText("Draw a line across the merged parasites first.")
            return
        if not self._label_ids:
            return

        from scipy.ndimage import distance_transform_edt
        from scipy.ndimage import label as ndlabel

        lid = self._label_ids[self._current_idx]
        mask = self._labels == lid

        # ── Convert stroke from thumbnail → image coordinates ─────────────────
        y0, x0, y1, x1 = self._thumb_crop
        crop_h, crop_w = y1 - y0, x1 - x0
        if crop_h <= 0 or crop_w <= 0:
            return

        scale = min(_THUMB_DISPLAY / crop_h, _THUMB_DISPLAY / crop_w)
        pix_h = int(crop_h * scale)
        pix_w = int(crop_w * scale)
        off_x = (_LABEL_SIZE - pix_w) / 2
        off_y = (_LABEL_SIZE - pix_h) / 2

        def _thumb_to_img(wx: int, wy: int) -> tuple[int, int]:
            ir = int((wy - off_y) / scale) + y0
            ic = int((wx - off_x) / scale) + x0
            return (
                max(0, min(self._labels.shape[0] - 1, ir)),
                max(0, min(self._labels.shape[1] - 1, ic)),
            )

        # Interpolate between consecutive stroke points so there are no gaps
        cut_pixels: set[tuple[int, int]] = set()
        prev = _thumb_to_img(*self._stroke_points[0])
        for wx, wy in self._stroke_points[1:]:
            cur = _thumb_to_img(wx, wy)
            for pt in _bresenham(prev[0], prev[1], cur[0], cur[1]):
                cut_pixels.add(pt)
            prev = cur

        # ── Remove cut pixels and find connected components ───────────────────
        temp_mask = mask.copy()
        for r, c in cut_pixels:
            if mask[r, c]:
                temp_mask[r, c] = False

        labeled_comp, n_comp = ndlabel(temp_mask)

        if n_comp < 2:
            self._split_info.setText(
                "Line didn't fully cross the segment — draw all the way through."
            )
            return

        # ── Merge tiny fragments into the nearest large component ─────────────
        mask_area = int(mask.sum())
        threshold = max(5, mask_area // 100)  # ignore fragments < 1% of mask
        comp_sizes = {k: int((labeled_comp == k).sum()) for k in range(1, n_comp + 1)}
        large = [k for k, s in comp_sizes.items() if s >= threshold]
        if len(large) < 2:
            self._split_info.setText(
                "Cut produced only one significant region — draw further across."
            )
            return

        # Build final label image: large components stay; unassigned pixels
        # (small fragments + cut pixels within mask) go to nearest large component.
        final = np.zeros_like(labeled_comp)
        for k in large:
            final[labeled_comp == k] = k

        unassigned = mask & (final == 0)
        if unassigned.any():
            _, (nr_arr, nc_arr) = distance_transform_edt(
                final == 0, return_indices=True
            )
            final[unassigned] = final[nr_arr[unassigned], nc_arr[unassigned]]

        final[~mask] = 0
        unique_ids = [u for u in np.unique(final) if u != 0]

        # ── Assign new label IDs ──────────────────────────────────────────────
        current_max = int(self._labels.max())
        new_label_ids: list[int] = []
        for seg_id in unique_ids:
            current_max += 1
            self._labels[final == seg_id] = current_max
            new_label_ids.append(current_max)

        # Update label_ids list: remove old, insert new at same position.
        insert_pos = self._current_idx
        self._label_ids.pop(insert_pos)
        if lid in self._decisions:
            del self._decisions[lid]
        parent_vac = self._vacuole_assignments.pop(lid, insert_pos + 1)

        for i, nl in enumerate(new_label_ids):
            self._label_ids.insert(insert_pos + i, nl)
            self._decisions[nl] = -1
            # New segments inherit the parent's vacuole ID so grouping is preserved
            self._vacuole_assignments[nl] = parent_vac

        # Re-measure just the new segments so the info panel shows real values
        self._remeasure_labels(new_label_ids)

        # Exit split mode and navigate to the first new segment
        self._cancel_split()
        self._current_idx = min(insert_pos, len(self._label_ids) - 1)
        self._refresh()

    def _remeasure_labels(self, label_ids: list[int]):
        """Re-run measure_pvs for the given label IDs and update self._measurements."""
        import pandas as pd

        from ._measure import measure_pvs

        if self._measurements is None:
            return

        # Build a temporary label image containing only the new labels
        temp = np.zeros_like(self._labels)
        for nl in label_ids:
            temp[self._labels == nl] = nl

        try:
            new_rows = measure_pvs(
                labels=temp,
                image=self._image,
                ch_cptsa=self._ch_cptsa,
                ch_mcherry=self._ch_mcherry,
                ch_names=self._ch_names,
                pixel_size_um=self._pixel_size_um,
            )
        except Exception:
            return  # measurements unavailable for new segments — info panel shows "—"

        # Drop the old row for the parent label (already removed from _label_ids)
        # and append the new rows
        self._measurements = pd.concat([self._measurements, new_rows]).loc[
            lambda df: ~df.index.duplicated(keep="last")
        ]

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

        # Find sibling parasites sharing the same vacuole so they appear as
        # cyan outlines in the thumbnail for context.  Search all entries in
        # _vacuole_assignments (not just _label_ids) so that in batch mode,
        # where select_one_per_vacuole removes non-representative parasites from
        # measurements, the full vacuole mask is still shown.
        current_vac = self._vacuole_assignments.get(lid)
        siblings = [
            other
            for other, vac in self._vacuole_assignments.items()
            if other != lid and vac == current_vac
        ]

        # Generate thumbnail and store crop coords for split-mode coordinate mapping
        thumb, crop = _make_thumbnail(
            self._image,
            self._labels,
            lid,
            self._ch_cptsa,
            self._ch_mcherry,
            sibling_label_ids=siblings,
        )
        self._thumb_raw = thumb
        self._thumb_crop = crop
        self._thumbnail_label.setPixmap(_array_to_pixmap(thumb, _THUMB_DISPLAY))

        # Count all labels sharing this vacuole directly from _vacuole_assignments
        # (more reliable than measurements['parasites_per_vacuole'] which is stale
        # after splits and missing for freshly-remeasured segments).
        n_in_vac = 1 + len(siblings)

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
            decision = {1: "✓ Accepted", 0: "✗ Rejected", -1: "Pending"}[
                self._decisions[lid]
            ]
            self._info_label.setText(
                f"Area: {area:,} px   Ratio: {ratio_str}   In vacuole: {n_in_vac}\n{decision}"
            )
        else:
            self._info_label.setText("")

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
