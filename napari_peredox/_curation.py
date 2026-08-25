"""
_curation.py — PV curation widgets for the two-stage pipeline.

VacuoleCurationWidget  (Stage 1)
---------------------------------
Reviews whole-vacuole segmentation output.  One page per vacuole.

  [← Prev]   Vacuole 3 / 12   [Next →]
  ┌──────────────────────────────────────┐
  │  thumbnail (green=cpTSa, red=mCherry) │
  │  vacuole boundary = white overlay    │
  └──────────────────────────────────────┘
  Area: 4 521 px
  [✓ Accept]   [✗ Reject]   [Skip]
  [✏ Draw outline]            ← polygon draw
  Status / [💾 Save & retrain vacuole model]

Draw-outline workflow
---------------------
Click to place polygon vertices (cyan dots + lines).
Double-click closes and fills the polygon — this replaces the vacuole mask.
Overlapping pixels are subtracted from neighbouring vacuoles (option B).
"Clear drawing" resets the current polygon.

CurationWidget  (Stage 2)
--------------------------
Reviews individual parasites grouped by vacuole.  One page = one vacuole,
showing all its parasites at once.

  [← Prev vacuole]   Vacuole 3 / 12   [Next →]
  ┌────────────────────────────────────────────┐
  │  thumbnail: all parasites in this vacuole  │
  │  selected parasite = white, others = cyan  │
  └────────────────────────────────────────────┘
  Parasite list (scrollable):
    P1  area 234 px  ratio 1.42   [✓][✗][✏]
    P2  area 198 px  ratio 1.38   [✓][✗][✏]
  [✓ Accept all]  [✗ Reject all]
  Status / [💾 Save & retrain parasite model]

Clicking [✏] on a row enters polygon draw mode for that parasite.
The filled polygon replaces its pixels; overlap subtracted from neighbours.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    pass

_THUMB_DISPLAY = 260
_LABEL_SIZE = 260  # match pixmap size so event.x/y() map 1:1 with no centering offset


# ---------------------------------------------------------------------------
# Shared image helpers
# ---------------------------------------------------------------------------


def _array_to_pixmap(rgb: np.ndarray, size: int = _THUMB_DISPLAY) -> QPixmap:
    """Convert (H, W, 3) uint8 array to a scaled QPixmap."""
    h, w = rgb.shape[:2]
    qimg = QImage(rgb.tobytes(), w, h, w * 3, QImage.Format_RGB888)
    pix = QPixmap.fromImage(qimg)
    return pix.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)


def _make_thumbnail(
    image: np.ndarray,
    labels: np.ndarray,
    focus_ids: list[int],
    other_ids: list[int],
    ch_cptsa: int,
    ch_mcherry: int,
    pad: int = 24,
    crop_ids: list[int] | None = None,
    highlight_ids: list[int] | None = None,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """
    Build an (H, W, 3) uint8 RGB thumbnail.

    focus_ids outlines → white; other_ids outlines → cyan.
    highlight_ids → yellow filled overlay (40% blend) to indicate selection.
    crop_ids controls the bounding box (defaults to focus_ids when None).
    Returns (rgb, (y0, x0, y1, x1)) crop coordinates.
    """
    from skimage.segmentation import find_boundaries

    # Crop bounds from crop_ids (or focus_ids) — don't let other_ids expand the view
    bound_ids = crop_ids if crop_ids is not None else focus_ids
    ys_focus, xs_focus = [], []
    for lid in bound_ids:
        ys, xs = np.where(labels == lid)
        if len(ys):
            ys_focus.extend(ys.tolist())
            xs_focus.extend(xs.tolist())

    if not ys_focus:
        return np.zeros((64, 64, 3), dtype=np.uint8), (0, 0, 64, 64)

    ys_all, xs_all = ys_focus, xs_focus

    y0 = max(int(min(ys_all)) - pad, 0)
    y1 = min(int(max(ys_all)) + pad + 1, labels.shape[0])
    x0 = max(int(min(xs_all)) - pad, 0)
    x1 = min(int(max(xs_all)) + pad + 1, labels.shape[1])

    if image.ndim == 2:
        image = image[..., np.newaxis]
    n_ch = image.shape[-1]

    def _norm(ch: int) -> np.ndarray:
        if ch >= n_ch:
            return np.zeros((y1 - y0, x1 - x0), dtype=np.float32)
        crop = image[y0:y1, x0:x1, ch].astype(np.float32)
        mn, mx = crop.min(), crop.max()
        return (crop - mn) / (mx - mn + 1e-9)

    green = _norm(ch_cptsa)
    red = _norm(ch_mcherry)
    rgb = np.stack([red, green, np.zeros_like(green)], axis=-1)

    # Yellow filled overlay for highlighted (selected) parasites
    if highlight_ids:
        yellow = np.array([1.0, 1.0, 0.0], dtype=np.float32)
        for lid in highlight_ids:
            m = labels[y0:y1, x0:x1] == lid
            if m.any():
                rgb[m] = rgb[m] * 0.6 + yellow * 0.4

    for lid in other_ids:
        m = labels[y0:y1, x0:x1] == lid
        if m.any():
            rgb[find_boundaries(m, mode="outer")] = [0.0, 0.8, 1.0]

    for lid in focus_ids:
        m = labels[y0:y1, x0:x1] == lid
        if m.any():
            rgb[find_boundaries(m, mode="outer")] = [1.0, 1.0, 1.0]

    return (rgb * 255).clip(0, 255).astype(np.uint8), (y0, x0, y1, x1)


def _thumb_to_image_coords(
    wx: int,
    wy: int,
    crop: tuple[int, int, int, int],
    label_size: int = _LABEL_SIZE,
    thumb_display: int = _THUMB_DISPLAY,
    img_shape: tuple[int, int] = (0, 0),
) -> tuple[int, int]:
    """Map thumbnail widget pixel → image pixel coordinates.

    _array_to_pixmap scales the crop to fit thumb_display×thumb_display
    with KeepAspectRatio, then Qt centers the result in the label_size×label_size
    QLabel.  We recompute the exact same scaled dimensions Qt uses so our
    offset math matches pixel-for-pixel.
    """
    y0, x0, y1, x1 = crop
    crop_h, crop_w = y1 - y0, x1 - x0
    if crop_h <= 0 or crop_w <= 0:
        return 0, 0
    # Qt KeepAspectRatio: scale so the larger dimension equals thumb_display.
    # Use the same integer arithmetic Qt applies internally.
    scale_h = thumb_display / crop_h
    scale_w = thumb_display / crop_w
    scale = min(scale_h, scale_w)
    # Qt rounds scaled dimensions to integers
    pix_h = round(crop_h * scale)
    pix_w = round(crop_w * scale)
    # AlignCenter padding inside the label widget
    off_x = (label_size - pix_w) / 2
    off_y = (label_size - pix_h) / 2
    ir = int((wy - off_y) / scale) + y0
    ic = int((wx - off_x) / scale) + x0
    H, W = img_shape
    ir = max(0, min(H - 1, ir)) if H > 0 else ir
    ic = max(0, min(W - 1, ic)) if W > 0 else ic
    return ir, ic


# ---------------------------------------------------------------------------
# Polygon draw thumbnail widget (shared)
# ---------------------------------------------------------------------------


class _PolygonThumbnail(QLabel):
    """
    QLabel that supports freehand drawing to define a mask outline.

    Press and drag to draw — points are sampled as the mouse moves.
    Release the button to close the shape and emit the filled polygon.
    Emits polygon_closed(list_of_image_coords) when the shape is finalised.
    """

    polygon_closed = Signal(object)  # list[tuple[int,int]] in image coords

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setFixedSize(_LABEL_SIZE, _LABEL_SIZE)
        self.setStyleSheet("background-color: black;")
        self.setCursor(Qt.CrossCursor)

        self._vertices: list[tuple[int, int]] = []  # widget coords
        self._drawing_now = False  # True while mouse button is held
        self._thumb_raw: np.ndarray | None = None
        self._crop: tuple[int, int, int, int] | None = None
        self._img_shape: tuple[int, int] = (0, 0)
        self._active = False

    def start(
        self,
        thumb_raw: np.ndarray,
        crop: tuple[int, int, int, int],
        img_shape: tuple[int, int],
    ) -> None:
        self._thumb_raw = thumb_raw
        self._crop = crop
        self._img_shape = img_shape
        self._vertices = []
        self._drawing_now = False
        self._active = True
        self._redraw()

    def stop(self) -> None:
        self._active = False
        self._drawing_now = False
        self._vertices = []
        if self._thumb_raw is not None:
            self.setPixmap(_array_to_pixmap(self._thumb_raw, _THUMB_DISPLAY))

    def clear_polygon(self) -> None:
        self._vertices = []
        self._drawing_now = False
        self._redraw()

    def mousePressEvent(self, event) -> None:
        if not self._active:
            super().mousePressEvent(event)
            return
        if event.button() == Qt.LeftButton:
            self._vertices = [(event.x(), event.y())]
            self._drawing_now = True
            self._redraw()
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        if not self._active or not self._drawing_now:
            super().mouseMoveEvent(event)
            return
        # Sample every ~3px to keep the point count manageable
        if self._vertices:
            lx, ly = self._vertices[-1]
            dx, dy = event.x() - lx, event.y() - ly
            if dx * dx + dy * dy < 9:
                event.accept()
                return
        self._vertices.append((event.x(), event.y()))
        self._redraw()
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        if not self._active or not self._drawing_now:
            super().mouseReleaseEvent(event)
            return
        if event.button() == Qt.LeftButton:
            self._drawing_now = False
            if len(self._vertices) >= 3:
                self._close_polygon()
            else:
                self.clear_polygon()
        event.accept()

    def mouseDoubleClickEvent(self, event) -> None:
        # Consume double-clicks to prevent napari's double_click_to_zoom
        event.accept()

    def _close_polygon(self) -> None:
        if len(self._vertices) < 3 or self._crop is None:
            return
        img_coords = [
            _thumb_to_image_coords(
                wx,
                wy,
                self._crop,
                _LABEL_SIZE,
                _THUMB_DISPLAY,
                self._img_shape,
            )
            for wx, wy in self._vertices
        ]
        self.polygon_closed.emit(img_coords)
        self.stop()

    def _redraw(self) -> None:
        if self._thumb_raw is None:
            return
        pix = _array_to_pixmap(self._thumb_raw, _THUMB_DISPLAY)
        if not self._vertices:
            self.setPixmap(pix)
            return
        painter = QPainter(pix)
        painter.setRenderHint(QPainter.Antialiasing)
        pen = QPen(QColor(0, 220, 255), 2)
        painter.setPen(pen)
        for i in range(1, len(self._vertices)):
            x0, y0 = self._vertices[i - 1]
            x1, y1 = self._vertices[i]
            painter.drawLine(x0, y0, x1, y1)
        # Close the loop with a dashed line back to start
        if len(self._vertices) >= 2:
            from qtpy.QtCore import Qt as _Qt

            pen_close = QPen(QColor(0, 220, 255), 1)
            pen_close.setStyle(_Qt.DashLine)
            painter.setPen(pen_close)
            x0, y0 = self._vertices[-1]
            x1, y1 = self._vertices[0]
            painter.drawLine(x0, y0, x1, y1)
        painter.end()
        self.setPixmap(pix)


def _polygon_to_mask(
    img_coords: list[tuple[int, int]],
    shape: tuple[int, int],
) -> np.ndarray:
    """
    Rasterise a polygon (list of (row, col) image coordinates) to a bool mask.
    Uses skimage.draw.polygon for anti-aliased fill.
    """
    from skimage.draw import polygon as sk_polygon

    rows = [r for r, c in img_coords]
    cols = [c for r, c in img_coords]
    rr, cc = sk_polygon(rows, cols, shape)
    mask = np.zeros(shape, dtype=bool)
    mask[rr, cc] = True
    return mask


def _sync_layer_data(layer, data: np.ndarray, viewer=None) -> None:
    """
    Update a napari Labels layer's pixel values without resetting the camera.

    Replacing layer.data triggers napari's _on_data_set → reset_view chain,
    which zooms out to fit the full image.  In-place modification of the
    existing array avoids that signal entirely — napari sees the pixels change
    but does not recalculate the camera extent.
    """
    try:
        if layer.data.shape == data.shape and layer.data.dtype == data.dtype:
            layer.data[:] = data
            layer.refresh()
            return
    except Exception:
        pass
    # Shape/dtype mismatch — fall back to full replacement with camera restore
    if viewer is None:
        layer.data = data
        return
    try:
        cam = viewer.camera
        center = tuple(cam.center)
        zoom = float(cam.zoom)
        angles = tuple(cam.angles)
        layer.data = data
        cam.center = center
        cam.zoom = zoom
        cam.angles = angles
    except Exception:
        layer.data = data


# ---------------------------------------------------------------------------
# Vacuole curation widget (Stage 1)
# ---------------------------------------------------------------------------


class VacuoleCurationWidget(QWidget):
    """
    Simple curation panel for Stage 1 (whole-vacuole) segmentation.

    Each page shows one vacuole with accept / reject buttons and a
    polygon-draw tool that lets the user redraw the vacuole boundary.
    """

    curation_saved = Signal()

    def __init__(
        self,
        vac_labels: np.ndarray,
        image: np.ndarray,
        ch_cptsa: int,
        ch_mcherry: int,
        on_save: Callable | None = None,
        viewer=None,
        labels_layer_name: str | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self._labels = vac_labels.copy()
        self._image = image
        self._ch_cptsa = ch_cptsa
        self._ch_mcherry = ch_mcherry
        self._on_save = on_save
        self._viewer = viewer
        self._labels_layer_name = labels_layer_name

        self._vac_ids: list[int] = sorted(
            int(v) for v in np.unique(self._labels) if v != 0
        )
        self._decisions: dict[int, int] = {v: -1 for v in self._vac_ids}
        self._current_idx = 0
        self._drawing = False

        self._build_ui()
        self._refresh()

    # ── UI ───────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Navigation
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

        # Thumbnail with polygon support
        self._thumb = _PolygonThumbnail()
        self._thumb.polygon_closed.connect(self._on_polygon_closed)
        layout.addWidget(self._thumb, alignment=Qt.AlignHCenter)

        # Info
        self._info_label = QLabel()
        self._info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._info_label)

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

        # Polygon draw controls
        self._btn_draw = QPushButton("✏  Draw outline")
        self._btn_draw.setToolTip(
            "Hold and drag to draw a freehand outline — releases to apply.\n"
            "Overlapping pixels are subtracted from neighbours."
        )
        self._btn_draw.clicked.connect(self._start_draw)
        layout.addWidget(self._btn_draw)

        self._draw_panel = QWidget()
        dp_layout = QVBoxLayout(self._draw_panel)
        dp_layout.setContentsMargins(0, 0, 0, 0)
        dp_layout.setSpacing(4)
        self._draw_info = QLabel("Click ✏ Draw outline to redraw the vacuole boundary.")
        self._draw_info.setAlignment(Qt.AlignCenter)
        self._draw_info.setStyleSheet("color: #888888; font-style: italic;")
        self._draw_info.setWordWrap(True)
        dp_layout.addWidget(self._draw_info)
        self._btn_clear_poly = QPushButton("Clear drawing")
        self._btn_clear_poly.clicked.connect(self._thumb.clear_polygon)
        dp_layout.addWidget(self._btn_clear_poly)
        self._btn_cancel_draw = QPushButton("Cancel")
        self._btn_cancel_draw.clicked.connect(self._cancel_draw)
        dp_layout.addWidget(self._btn_cancel_draw)
        self._draw_panel.setEnabled(False)  # greyed out until draw mode active
        layout.addWidget(self._draw_panel)

        # Status
        self._status_label = QLabel()
        self._status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._status_label)

        self._btn_save = QPushButton("💾 Save & retrain vacuole model")
        self._btn_save.clicked.connect(self._save)
        layout.addWidget(self._btn_save)

        layout.addStretch()

    # ── Navigation ───────────────────────────────────────────────────────────

    def _prev(self) -> None:
        self._cancel_draw()
        if self._current_idx > 0:
            self._current_idx -= 1
        self._refresh()

    def _next(self) -> None:
        self._cancel_draw()
        if self._current_idx < len(self._vac_ids) - 1:
            self._current_idx += 1
        self._refresh()

    def _accept(self) -> None:
        self._cancel_draw()
        if self._vac_ids:
            self._decisions[self._vac_ids[self._current_idx]] = 1
            if self._current_idx < len(self._vac_ids) - 1:
                self._current_idx += 1
        self._refresh()

    def _reject(self) -> None:
        self._cancel_draw()
        if self._vac_ids:
            self._decisions[self._vac_ids[self._current_idx]] = 0
            if self._current_idx < len(self._vac_ids) - 1:
                self._current_idx += 1
        self._refresh()

    def _save(self) -> None:
        # Treat any unreviewed (pending) vacuoles as accepted so the user only
        # needs to explicitly reject bad ones — not approve every single one.
        final = {vid: (dec if dec != -1 else 1) for vid, dec in self._decisions.items()}
        if self._on_save:
            self._on_save(final, self._labels)
        self.curation_saved.emit()

    # ── Polygon draw ─────────────────────────────────────────────────────────

    def _start_draw(self) -> None:
        if not self._vac_ids:
            return
        self._drawing = True
        self._draw_panel.setEnabled(True)
        self._draw_info.setText("Hold and drag to draw outline.  Release to apply.")
        self._draw_info.setStyleSheet("color: #00dcff; font-style: italic;")
        self._btn_draw.setEnabled(False)
        vac_id = self._vac_ids[self._current_idx]
        others = [v for v in self._vac_ids if v != vac_id]
        thumb, crop = _make_thumbnail(
            self._image,
            self._labels,
            [vac_id],
            others,
            self._ch_cptsa,
            self._ch_mcherry,
        )
        self._thumb.start(thumb, crop, self._labels.shape[:2])

    def _cancel_draw(self) -> None:
        if not self._drawing:
            return
        self._drawing = False
        self._thumb.stop()
        self._draw_panel.setEnabled(False)
        self._draw_info.setText("Click ✏ Draw outline to redraw the vacuole boundary.")
        self._draw_info.setStyleSheet("color: #888888; font-style: italic;")
        self._btn_draw.setEnabled(True)

    def _on_polygon_closed(self, img_coords: list) -> None:
        """Apply the drawn polygon as the new mask for the current vacuole."""
        self._drawing = False
        self._draw_panel.setEnabled(False)
        self._draw_info.setText("Click ✏ Draw outline to redraw the vacuole boundary.")
        self._draw_info.setStyleSheet("color: #888888; font-style: italic;")
        self._btn_draw.setEnabled(True)

        if not self._vac_ids:
            return
        vac_id = self._vac_ids[self._current_idx]
        new_mask = _polygon_to_mask(img_coords, self._labels.shape[:2])

        # Option B: subtract new pixels from all other vacuoles first
        conflict = new_mask & (self._labels != 0) & (self._labels != vac_id)
        self._labels[conflict] = 0

        # Replace this vacuole's pixels
        self._labels[self._labels == vac_id] = 0
        self._labels[new_mask] = vac_id

        # Sync napari layer — preserve camera so the view doesn't jump
        layer = self._get_labels_layer()
        if layer is not None:
            _sync_layer_data(layer, self._labels, self._viewer)

        self._refresh()

    def _get_labels_layer(self):
        if self._viewer is None:
            return None
        name = self._labels_layer_name
        if name and name in self._viewer.layers:
            return self._viewer.layers[name]
        from napari.layers import Labels

        for layer in self._viewer.layers:
            if isinstance(layer, Labels):
                return layer
        return None

    # ── Display ──────────────────────────────────────────────────────────────

    def _refresh(self) -> None:
        n = len(self._vac_ids)
        if n == 0:
            self._lbl_nav.setText("No vacuoles")
            self._info_label.setText("")
            self._thumb.stop()
            self._update_status()
            return

        idx = self._current_idx
        vac_id = self._vac_ids[idx]
        self._lbl_nav.setText(f"Vacuole {idx + 1} / {n}  (id={vac_id})")

        others = [v for v in self._vac_ids if v != vac_id]
        thumb, _crop = _make_thumbnail(
            self._image,
            self._labels,
            [vac_id],
            others,
            self._ch_cptsa,
            self._ch_mcherry,
        )
        self._thumb._thumb_raw = thumb
        self._thumb.setPixmap(_array_to_pixmap(thumb, _THUMB_DISPLAY))

        area = int((self._labels == vac_id).sum())
        dec_str = {1: "✓ Accepted", 0: "✗ Rejected", -1: "Pending"}[
            self._decisions[vac_id]
        ]
        self._info_label.setText(f"Area: {area:,} px   {dec_str}")
        self._update_status()

    def _update_status(self) -> None:
        n_a = sum(v == 1 for v in self._decisions.values())
        n_r = sum(v == 0 for v in self._decisions.values())
        n_p = sum(v == -1 for v in self._decisions.values())
        self._status_label.setText(
            f"✓ {n_a} accepted   ✗ {n_r} rejected   … {n_p} pending"
        )

    # ── Public ───────────────────────────────────────────────────────────────

    @property
    def decisions(self) -> dict[int, int]:
        return dict(self._decisions)

    @property
    def curated_labels(self) -> np.ndarray:
        """Vacuole label array after polygon edits."""
        return self._labels


# ---------------------------------------------------------------------------
# Helper: _ClickableThumbnail for split tool (kept for ParasiteCurationWidget)
# ---------------------------------------------------------------------------


def _bresenham(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
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


# ---------------------------------------------------------------------------
# Parasite curation widget (Stage 2) — per-vacuole gallery
# ---------------------------------------------------------------------------


class CurationWidget(QWidget):
    """
    Stage 2 curation: reviews individual parasites grouped by vacuole.

    One page = one vacuole.  All parasites in that vacuole are shown together
    with per-row accept / reject / draw-outline controls.
    """

    curation_saved = Signal()

    def __init__(
        self,
        labels: np.ndarray,
        image: np.ndarray,
        measurements,
        ch_cptsa: int,
        ch_mcherry: int,
        ch_names: dict[int, str] | None = None,
        pixel_size_um: float | None = None,
        vacuole_assignments: dict[int, int] | None = None,
        accepted_vac_ids: list[int] | None = None,
        vac_labels: np.ndarray | None = None,
        on_save: Callable | None = None,
        viewer=None,
        labels_layer_name: str | None = None,
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
        self._viewer = viewer
        self._labels_layer_name = labels_layer_name
        self._editing_label: int | None = None  # which parasite is being drawn
        self._add_para_mode: bool = False  # True when adding a brand-new parasite
        self._selected_para: int | None = None  # highlighted in thumbnail
        self._vac_labels = vac_labels  # Stage 1 labels — used to crop empty vacuoles

        # Build vacuole → [parasite_labels] mapping
        if vacuole_assignments:
            self._vac_map: dict[int, int] = dict(vacuole_assignments)
        elif measurements is not None and "vacuole_id" in measurements.columns:
            self._vac_map = {
                int(lbl): int(measurements.loc[lbl, "vacuole_id"])
                for lbl in measurements.index
                if lbl in measurements.index
            }
        else:
            all_lbls = (
                measurements.index.tolist()
                if measurements is not None and len(measurements) > 0
                else []
            )
            self._vac_map = {lbl: i + 1 for i, lbl in enumerate(all_lbls)}

        # Group by vacuole
        from collections import defaultdict

        vac_to_para: dict[int, list[int]] = defaultdict(list)
        for para, vac in self._vac_map.items():
            vac_to_para[vac].append(para)
        self._vac_to_para: dict[int, list[int]] = dict(vac_to_para)

        # Seed vacuoles that had zero parasites detected so they still appear
        # in the gallery and the user can annotate missed parasites via "+ Add".
        if accepted_vac_ids:
            for vid in accepted_vac_ids:
                if vid not in self._vac_to_para:
                    self._vac_to_para[vid] = []

        self._vac_ids: list[int] = sorted(self._vac_to_para.keys())
        self._current_vac_idx = 0

        # Per-parasite decisions
        all_para_ids = list(self._vac_map.keys())
        self._decisions: dict[int, int] = {pid: -1 for pid in all_para_ids}

        # In-place update dicts (avoid layout rebuild on per-row changes)
        self._row_labels: dict[int, QLabel] = {}
        self._row_widgets: dict[int, QWidget] = {}

        self._build_ui()
        self._refresh()

    # ── UI ───────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Navigation
        nav = QHBoxLayout()
        self._btn_prev = QPushButton("← Prev vacuole")
        self._btn_prev.clicked.connect(self._prev_vac)
        self._lbl_nav = QLabel()
        self._lbl_nav.setAlignment(Qt.AlignCenter)
        self._btn_next = QPushButton("Next vacuole →")
        self._btn_next.clicked.connect(self._next_vac)
        nav.addWidget(self._btn_prev)
        nav.addWidget(self._lbl_nav, stretch=1)
        nav.addWidget(self._btn_next)
        layout.addLayout(nav)

        # Thumbnail
        self._thumb = _PolygonThumbnail()
        self._thumb.polygon_closed.connect(self._on_polygon_closed)
        layout.addWidget(self._thumb, alignment=Qt.AlignHCenter)

        # Parasite list (scrollable)
        self._list_scroll = QScrollArea()
        self._list_scroll.setWidgetResizable(True)
        self._list_scroll.setFixedHeight(160)
        self._list_container = QWidget()
        self._list_layout = QVBoxLayout(self._list_container)
        self._list_layout.setContentsMargins(2, 2, 2, 2)
        self._list_layout.setSpacing(2)
        self._list_scroll.setWidget(self._list_container)
        layout.addWidget(self._list_scroll)

        # Add missed parasite
        self._btn_add_para = QPushButton("+ Add missed parasite")
        self._btn_add_para.setToolTip(
            "Draw a freehand outline for a parasite the segmentation missed."
        )
        self._btn_add_para.clicked.connect(self._start_add_para)
        layout.addWidget(self._btn_add_para)

        # Polygon draw panel — always visible, enabled only in draw mode
        self._draw_panel = QWidget()
        dp = QVBoxLayout(self._draw_panel)
        dp.setContentsMargins(0, 0, 0, 0)
        dp.setSpacing(3)
        self._draw_info = QLabel("Click ✏ on a parasite row to redraw its outline.")
        self._draw_info.setAlignment(Qt.AlignCenter)
        self._draw_info.setStyleSheet("color: #888888; font-style: italic;")
        self._draw_info.setWordWrap(True)
        dp.addWidget(self._draw_info)
        draw_btns = QHBoxLayout()
        self._btn_clear_poly = QPushButton("Clear")
        self._btn_clear_poly.clicked.connect(self._thumb.clear_polygon)
        self._btn_cancel_draw = QPushButton("Cancel")
        self._btn_cancel_draw.clicked.connect(self._cancel_draw)
        draw_btns.addWidget(self._btn_clear_poly)
        draw_btns.addWidget(self._btn_cancel_draw)
        dp.addLayout(draw_btns)
        self._draw_panel.setEnabled(False)  # greyed out until draw mode active
        layout.addWidget(self._draw_panel)

        # Bulk actions
        bulk = QHBoxLayout()
        self._btn_accept_all = QPushButton("✓ Accept all")
        self._btn_accept_all.setStyleSheet("background-color: #2e7d32; color: white;")
        self._btn_accept_all.clicked.connect(self._accept_all)
        self._btn_reject_all = QPushButton("✗ Reject all")
        self._btn_reject_all.setStyleSheet("background-color: #c62828; color: white;")
        self._btn_reject_all.clicked.connect(self._reject_all)
        bulk.addWidget(self._btn_accept_all)
        bulk.addWidget(self._btn_reject_all)
        layout.addLayout(bulk)

        # Status + save
        self._status_label = QLabel()
        self._status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._status_label)

        self._btn_save = QPushButton("💾 Save & retrain parasite model")
        self._btn_save.clicked.connect(self._save)
        layout.addWidget(self._btn_save)

        layout.addStretch()

    # ── Navigation ───────────────────────────────────────────────────────────

    def _prev_vac(self) -> None:
        self._cancel_draw()
        self._selected_para = None
        if self._current_vac_idx > 0:
            self._current_vac_idx -= 1
        self._refresh()

    def _next_vac(self) -> None:
        self._cancel_draw()
        self._selected_para = None
        if self._current_vac_idx < len(self._vac_ids) - 1:
            self._current_vac_idx += 1
        self._refresh()

    def _accept_all(self) -> None:
        self._cancel_draw()
        if not self._vac_ids:
            return
        vac_id = self._vac_ids[self._current_vac_idx]
        for pid in self._vac_to_para.get(vac_id, []):
            self._decisions[pid] = 1
        self._refresh()

    def _reject_all(self) -> None:
        self._cancel_draw()
        if not self._vac_ids:
            return
        vac_id = self._vac_ids[self._current_vac_idx]
        for pid in self._vac_to_para.get(vac_id, []):
            self._decisions[pid] = 0
        self._refresh()

    def _save(self) -> None:
        if self._on_save:
            self._on_save(self._decisions, self._vac_map)
        self.curation_saved.emit()

    # ── Per-parasite row actions ──────────────────────────────────────────────

    def _select_para(self, pid: int) -> None:
        """Toggle yellow highlight on a parasite in the thumbnail."""
        self._selected_para = pid if self._selected_para != pid else None
        self._update_thumbnail()
        self._update_row_styles()

    def _accept_para(self, pid: int) -> None:
        self._decisions[pid] = 1
        self._update_row_label(pid)
        self._update_status()

    def _reject_para(self, pid: int) -> None:
        self._decisions[pid] = 0
        self._update_row_label(pid)
        self._update_status()

    def _draw_para(self, pid: int) -> None:
        """Enter freehand-draw mode for a specific parasite."""
        self._cancel_draw()
        self._editing_label = pid
        self._draw_panel.setEnabled(True)
        self._draw_info.setText("Hold and drag to draw outline.  Release to apply.")
        self._draw_info.setStyleSheet("color: #00dcff; font-style: italic;")

        vac_id = self._vac_ids[self._current_vac_idx]
        all_para = self._vac_to_para.get(vac_id, [])
        siblings = [p for p in all_para if p != pid]
        # Crop around the full vacuole (all parasites) for context.
        # Target parasite boundary = white; siblings = cyan.
        thumb, crop = _make_thumbnail(
            self._image,
            self._labels,
            [pid],
            siblings,
            self._ch_cptsa,
            self._ch_mcherry,
            crop_ids=all_para if all_para else [pid],
        )
        self._thumb.start(thumb, crop, self._labels.shape[:2])

    def _start_add_para(self) -> None:
        """Enter draw mode for a brand-new parasite (missed by segmentation)."""
        if not self._vac_ids:
            return
        self._cancel_draw()
        new_id = int(self._labels.max()) + 1
        self._editing_label = new_id
        self._add_para_mode = True
        self._draw_panel.setEnabled(True)
        self._draw_info.setText(
            f"Drawing new parasite P{new_id} — hold and drag, release to apply."
        )
        self._draw_info.setStyleSheet("color: #00dcff; font-style: italic;")
        # Show whole vacuole as context.
        # For empty vacuoles use _vac_labels to crop to the vacuole region.
        vac_id = self._vac_ids[self._current_vac_idx]
        all_para = self._vac_to_para.get(vac_id, [])
        if all_para:
            crop_ids = all_para
            label_source = self._labels
        elif self._vac_labels is not None:
            # No parasites yet — use the vacuole mask pixels as the crop anchor
            crop_ids = [vac_id]
            label_source = self._vac_labels
        else:
            crop_ids = None
            label_source = self._labels
        thumb, crop = _make_thumbnail(
            self._image,
            label_source,
            [],
            all_para,
            self._ch_cptsa,
            self._ch_mcherry,
            crop_ids=crop_ids,
        )
        self._thumb.start(thumb, crop, self._labels.shape[:2])

    def _cancel_draw(self) -> None:
        self._editing_label = None
        self._add_para_mode = False
        self._thumb.stop()
        self._draw_panel.setEnabled(False)
        self._draw_info.setText("Click ✏ on a parasite row to redraw its outline.")
        self._draw_info.setStyleSheet("color: #888888; font-style: italic;")

    def _on_polygon_closed(self, img_coords: list) -> None:
        """Apply drawn polygon as the new or updated mask for the edited parasite."""
        pid = self._editing_label
        adding = self._add_para_mode
        self._editing_label = None
        self._add_para_mode = False
        self._draw_panel.setEnabled(False)
        self._draw_info.setText("Click ✏ on a parasite row to redraw its outline.")
        self._draw_info.setStyleSheet("color: #888888; font-style: italic;")

        if pid is None:
            return
        new_mask = _polygon_to_mask(img_coords, self._labels.shape[:2])
        if not new_mask.any():
            return

        # Subtract new pixels from any other labels; clear old pixels for pid
        conflict = new_mask & (self._labels != 0) & (self._labels != pid)
        self._labels[conflict] = 0
        self._labels[self._labels == pid] = 0
        self._labels[new_mask] = pid

        if adding:
            # Register the new parasite in this vacuole
            vac_id = self._vac_ids[self._current_vac_idx]
            self._vac_to_para.setdefault(vac_id, []).append(pid)
            self._vac_map[pid] = vac_id
            self._decisions[pid] = 1  # auto-accept new additions

        # Remeasure affected label
        self._remeasure_labels([pid])

        # Sync napari layer in-place
        layer = self._get_labels_layer()
        if layer is not None:
            _sync_layer_data(layer, self._labels, self._viewer)

        if adding:
            # Row list changed structurally — must rebuild
            self._refresh()
        else:
            # Update only the row text + thumbnail — no layout change
            self._update_row_label(pid)
            self._update_thumbnail()
            self._update_status()

    # ── Measurement helpers ───────────────────────────────────────────────────

    def _remeasure_labels(self, label_ids: list[int]) -> None:
        import pandas as pd

        from ._measure import measure_pvs

        if self._measurements is None:
            return
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
            return
        self._measurements = pd.concat([self._measurements, new_rows]).loc[
            lambda df: ~df.index.duplicated(keep="last")
        ]

    def _get_labels_layer(self):
        if self._viewer is None:
            return None
        name = self._labels_layer_name
        if name and name in self._viewer.layers:
            return self._viewer.layers[name]
        from napari.layers import Labels

        for layer in self._viewer.layers:
            if isinstance(layer, Labels):
                return layer
        return None

    # ── Display ──────────────────────────────────────────────────────────────

    def _row_label_text(self, pid: int) -> str:
        m_row = (
            self._measurements.loc[pid]
            if self._measurements is not None and pid in self._measurements.index
            else None
        )
        area_str = f"{int(m_row.get('area_px', 0)):,} px" if m_row is not None else "—"
        ratio_val = (
            m_row.get("ratio_cptsa_mcherry", float("nan"))
            if m_row is not None
            else float("nan")
        )
        ratio_str = f"{ratio_val:.3f}" if not np.isnan(ratio_val) else "—"
        dec_icon = {1: "✓", 0: "✗", -1: "·"}.get(self._decisions.get(pid, -1), "·")
        return f"{dec_icon} P{pid}  {area_str}  ratio {ratio_str}"

    def _update_row_label(self, pid: int) -> None:
        """Update just the text of an existing row label — no layout change."""
        lbl = self._row_labels.get(pid)
        if lbl is not None:
            lbl.setText(self._row_label_text(pid))

    def _update_row_styles(self) -> None:
        """Highlight the selected row background; clear all others."""
        for pid, row_w in self._row_widgets.items():
            if pid == self._selected_para:
                row_w.setStyleSheet(
                    "QWidget { background-color: #4d4400; border-radius: 3px; }"
                )
            else:
                row_w.setStyleSheet("")

    def _update_thumbnail(self) -> None:
        """Redraw the thumbnail for the current vacuole without touching layout."""
        if not self._vac_ids:
            return
        vac_id = self._vac_ids[self._current_vac_idx]
        para_ids = self._vac_to_para.get(vac_id, [])
        sel = self._selected_para
        highlight = [sel] if sel is not None and sel in para_ids else None
        if para_ids:
            # Crop around detected parasites, highlight selected
            thumb, _ = _make_thumbnail(
                self._image,
                self._labels,
                para_ids,
                [],
                self._ch_cptsa,
                self._ch_mcherry,
                highlight_ids=highlight,
            )
        elif self._vac_labels is not None:
            # Empty vacuole — crop to vacuole extent using Stage 1 labels
            thumb, _ = _make_thumbnail(
                self._image,
                self._vac_labels,
                [vac_id],
                [],
                self._ch_cptsa,
                self._ch_mcherry,
            )
        else:
            return
        self._thumb._thumb_raw = thumb
        self._thumb.setPixmap(_array_to_pixmap(thumb, _THUMB_DISPLAY))

    def _refresh(self) -> None:
        # Clear the parasite list and rebuild row widgets
        self._row_labels.clear()
        self._row_widgets.clear()
        for i in reversed(range(self._list_layout.count())):
            w = self._list_layout.itemAt(i).widget()
            if w:
                w.setParent(None)

        n = len(self._vac_ids)
        if n == 0:
            self._lbl_nav.setText("No vacuoles")
            self._update_status()
            return

        vac_id = self._vac_ids[self._current_vac_idx]
        self._lbl_nav.setText(
            f"Vacuole {self._current_vac_idx + 1} / {n}  (id={vac_id})"
        )

        para_ids = self._vac_to_para.get(vac_id, [])

        self._update_thumbnail()

        # Per-parasite rows
        for pid in para_ids:
            row_w = QWidget()
            row_w.setCursor(Qt.PointingHandCursor)
            row_l = QHBoxLayout(row_w)
            row_l.setContentsMargins(2, 1, 2, 1)
            row_l.setSpacing(4)

            info = QLabel(self._row_label_text(pid))
            info.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            row_l.addWidget(info, stretch=1)
            self._row_labels[pid] = info
            self._row_widgets[pid] = row_w

            for icon, slot in [
                ("✓", lambda _p=pid: self._accept_para(_p)),
                ("✗", lambda _p=pid: self._reject_para(_p)),
                ("✏", lambda _p=pid: self._draw_para(_p)),
            ]:
                b = QPushButton(icon)
                b.setFixedWidth(28)
                b.clicked.connect(slot)
                row_l.addWidget(b)

            # Clicking the row background (outside buttons) selects/deselects
            row_w.mouseReleaseEvent = lambda _e, _p=pid: self._select_para(_p)

            self._list_layout.addWidget(row_w)

        self._update_row_styles()

        self._update_status()

    def _update_status(self) -> None:
        n_a = sum(v == 1 for v in self._decisions.values())
        n_r = sum(v == 0 for v in self._decisions.values())
        n_p = sum(v == -1 for v in self._decisions.values())
        self._status_label.setText(
            f"✓ {n_a} accepted   ✗ {n_r} rejected   … {n_p} pending"
        )

    # ── Public ───────────────────────────────────────────────────────────────

    @property
    def decisions(self) -> dict[int, int]:
        return dict(self._decisions)


# ---------------------------------------------------------------------------
# napari entry point (placeholder)
# ---------------------------------------------------------------------------


def make_curation_widget(napari_viewer=None):
    w = QWidget()
    from qtpy.QtWidgets import QVBoxLayout

    lyt = QVBoxLayout(w)
    lbl = QLabel(
        "Run 'Stage 1: Detect vacuoles' first,\n"
        "then use 'Review vacuoles' to open the curation panel."
    )
    lbl.setAlignment(Qt.AlignCenter)
    lbl.setWordWrap(True)
    lyt.addWidget(lbl)
    return w
