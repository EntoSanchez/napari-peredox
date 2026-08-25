import numpy as np

from napari_peredox._host import measure_hosts


def _fixture():
    """
    30x30, 2-channel image.
    Host 1: rows 2-27, cols 2-13 (interior).   Host 2: rows 0-29, cols 16-29 (touches border).
    Parasite 1 (5x5) inside host 1.  Host 2 is uninfected.
    Channel 0 (cptsa) = 20 in hosts, 500 in parasite; channel 1 (mcherry) = 10 everywhere non-bg.
    """
    hosts = np.zeros((30, 30), dtype=np.int32)
    hosts[2:28, 2:14] = 1
    hosts[:, 16:30] = 2
    paras = np.zeros_like(hosts)
    paras[10:15, 5:10] = 1
    img = np.zeros((30, 30, 2), dtype=np.float32)
    img[..., 0][hosts > 0] = 20.0
    img[..., 1][hosts > 0] = 10.0
    img[..., 0][paras > 0] = 500.0
    return hosts, paras, img


def test_one_row_per_host_with_infection_flags():
    hosts, paras, img = _fixture()
    df = measure_hosts(
        hosts, paras, img, para_to_host={1: 1}, vac_to_host={7: 1}, dilation_px=2
    )
    assert sorted(df.index.tolist()) == [1, 2]
    assert df.index.name == "host_id"
    assert bool(df.loc[1, "infected"]) is True
    assert bool(df.loc[2, "infected"]) is False
    assert int(df.loc[1, "n_parasites"]) == 1
    assert int(df.loc[2, "n_parasites"]) == 0
    assert int(df.loc[1, "n_vacuoles"]) == 1
    assert df.loc[1, "parasite_area_px"] == 25.0


def test_parasite_pixels_plus_dilation_excluded_from_host_ratio():
    hosts, paras, img = _fixture()
    df = measure_hosts(
        hosts, paras, img, para_to_host={1: 1}, vac_to_host={}, dilation_px=2
    )
    # If any 500-valued parasite pixel leaked into host 1's cytosol, the mean
    # cptsa would exceed 20.  Ratio = 20/10 = 2 exactly when exclusion worked.
    assert abs(df.loc[1, "ratio_intden"] - 2.0) < 1e-6
    # Cytosol area shrank by MORE than the raw parasite area (dilation buffer)
    assert df.loc[1, "area_px"] < df.loc[1, "host_area_px_total"] - 25.0


def test_host_area_px_total_is_full_footprint():
    hosts, paras, img = _fixture()
    df = measure_hosts(hosts, paras, img, para_to_host={1: 1}, vac_to_host={})
    assert df.loc[1, "host_area_px_total"] == float(26 * 12)
    assert df.loc[2, "host_area_px_total"] == float(30 * 14)


def test_on_border_flag():
    hosts, paras, img = _fixture()
    df = measure_hosts(hosts, paras, img, para_to_host={}, vac_to_host={})
    assert bool(df.loc[1, "on_border"]) is False
    assert bool(df.loc[2, "on_border"]) is True


def test_fully_covered_host_kept_with_nan_ratio():
    hosts = np.zeros((10, 10), dtype=np.int32)
    hosts[2:6, 2:6] = 1
    paras = np.zeros_like(hosts)
    paras[2:6, 2:6] = 1  # parasite covers the entire host
    img = np.ones((10, 10, 2), dtype=np.float32)
    df = measure_hosts(hosts, paras, img, para_to_host={1: 1}, vac_to_host={})
    assert 1 in df.index
    assert bool(df.loc[1, "cytosol_empty"]) is True
    assert np.isnan(df.loc[1, "ratio_intden"])
    assert bool(df.loc[1, "infected"]) is True


def test_dilation_zero_excludes_exact_mask_only():
    hosts, paras, img = _fixture()
    df = measure_hosts(
        hosts, paras, img, para_to_host={1: 1}, vac_to_host={}, dilation_px=0
    )
    assert df.loc[1, "area_px"] == df.loc[1, "host_area_px_total"] - 25.0
