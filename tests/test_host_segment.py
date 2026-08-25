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
    labels[2:10, 2:10] = 1  # 64 px
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
    filtered, raw, stats = segment_host_cells(_img(), channel_index=1, min_area_px=10.0)
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
    np.testing.assert_array_equal(fake_model.last_input, img[..., 1])


def test_2d_image_accepted(fake_model):
    img2d = np.full((20, 20), 10.0, dtype=np.float32)
    filtered, _, _ = segment_host_cells(img2d, channel_index=0)
    assert filtered.shape == (20, 20)


def test_model_receives_cpsam_kwargs(fake_model):
    segment_host_cells(
        _img(),
        channel_index=1,
        diameter=120.0,
        flow_threshold=0.5,
        cellprob_threshold=-1.0,
    )
    assert fake_model.last_kwargs["diameter"] == 120.0
    assert fake_model.last_kwargs["flow_threshold"] == 0.5
    assert fake_model.last_kwargs["cellprob_threshold"] == -1.0
