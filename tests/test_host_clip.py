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
