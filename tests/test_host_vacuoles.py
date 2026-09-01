"""Vacuole-level reporting for host+parasites mode."""

import numpy as np
import pandas as pd

from napari_peredox._host import (
    host_vacuole_summary,
    measure_hosts,
    measure_vacuoles_in_hosts,
)


def _scene():
    """
    40x40, 2 channels.
    Host 1 (rows 2-19, cols 2-19) holds vacuole 1 with parasites 1 and 2.
    Host 2 (rows 22-38, cols 22-38) holds vacuole 2 with parasite 3.
    Vacuole lumen is brighter on cptsa than the host cytosol; parasite bodies
    brighter still, so lumen-vs-parasite ratios are distinguishable.
    """
    hosts = np.zeros((40, 40), dtype=np.int32)
    hosts[2:20, 2:20] = 1
    hosts[22:39, 22:39] = 2

    vacs = np.zeros_like(hosts)
    vacs[5:13, 5:13] = 1  # 64 px, inside host 1
    vacs[26:32, 26:32] = 2  # 36 px, inside host 2

    paras = np.zeros_like(hosts)
    paras[6:9, 6:9] = 1  # 9 px  \ both inside vacuole 1
    paras[9:12, 9:12] = 2  # 9 px /
    paras[27:31, 27:31] = 3  # 16 px, inside vacuole 2

    img = np.zeros((40, 40, 2), dtype=np.float32)
    img[..., 0][hosts > 0] = 10.0  # cptsa: cytosol
    img[..., 1][hosts > 0] = 10.0  # mcherry: flat everywhere -> ratio = cptsa/10
    img[..., 0][vacs > 0] = 30.0  # lumen brighter
    img[..., 1][vacs > 0] = 10.0
    img[..., 0][paras > 0] = 60.0  # parasite bodies brightest
    img[..., 1][paras > 0] = 10.0
    return hosts, vacs, paras, img


VAC_TO_HOST = {1: 1, 2: 2}
PARA_TO_HOST = {1: 1, 2: 1, 3: 2}
VACUOLE_MAP = {1: 1, 2: 1, 3: 2}  # parasite label -> vacuole id


def test_one_row_per_vacuole_with_host_and_counts():
    hosts, vacs, paras, img = _scene()
    df = measure_vacuoles_in_hosts(
        vac_labels=vacs,
        para_labels=paras,
        image=img,
        vac_to_host=VAC_TO_HOST,
        vacuole_map=VACUOLE_MAP,
    )
    assert sorted(df.index.tolist()) == [1, 2]
    assert df.index.name == "vacuole_id"
    assert df.loc[1, "host_id"] == 1
    assert df.loc[2, "host_id"] == 2
    assert int(df.loc[1, "parasites_per_vacuole"]) == 2
    assert int(df.loc[2, "parasites_per_vacuole"]) == 1


def test_vacuole_area_and_lumen_ratio_measured():
    hosts, vacs, paras, img = _scene()
    df = measure_vacuoles_in_hosts(
        vac_labels=vacs,
        para_labels=paras,
        image=img,
        vac_to_host=VAC_TO_HOST,
        vacuole_map=VACUOLE_MAP,
        pixel_size_um=0.5,
    )
    assert df.loc[1, "area_px"] == 64.0
    assert df.loc[2, "area_px"] == 36.0
    assert abs(df.loc[1, "area_um2"] - 16.0) < 1e-6  # 64 px * 0.25 um2/px
    # Vacuole 2 lumen: 20 px at 30 + 16 px at 60 -> intden 1560 / 360 mcherry
    assert abs(df.loc[2, "ratio_intden"] - (1560.0 / 360.0)) < 1e-6


def test_parasite_ratio_aggregates_per_vacuole():
    hosts, vacs, paras, img = _scene()
    df = measure_vacuoles_in_hosts(
        vac_labels=vacs,
        para_labels=paras,
        image=img,
        vac_to_host=VAC_TO_HOST,
        vacuole_map=VACUOLE_MAP,
    )
    # Every parasite body is 60/10 = 6.0
    assert abs(df.loc[1, "mean_parasite_ratio"] - 6.0) < 1e-6
    assert abs(df.loc[1, "median_parasite_ratio"] - 6.0) < 1e-6


def test_vacuole_with_no_assigned_host_is_dropped():
    hosts, vacs, paras, img = _scene()
    df = measure_vacuoles_in_hosts(
        vac_labels=vacs,
        para_labels=paras,
        image=img,
        vac_to_host={1: 1},
        vacuole_map=VACUOLE_MAP,  # vacuole 2 unassigned
    )
    assert df.index.tolist() == [1]


def test_empty_inputs_give_empty_frame():
    hosts, vacs, paras, img = _scene()
    df = measure_vacuoles_in_hosts(
        vac_labels=np.zeros_like(vacs),
        para_labels=paras,
        image=img,
        vac_to_host={},
        vacuole_map={},
    )
    assert df.empty


def test_host_summary_aggregates_vacuole_rows():
    hosts, vacs, paras, img = _scene()
    vdf = measure_vacuoles_in_hosts(
        vac_labels=vacs,
        para_labels=paras,
        image=img,
        vac_to_host=VAC_TO_HOST,
        vacuole_map=VACUOLE_MAP,
    )
    s = host_vacuole_summary(vdf, host_ids=[1, 2, 3])
    assert list(s.index) == [1, 2, 3]
    assert s.loc[1, "vacuole_area_px_total"] == 64.0
    assert s.loc[1, "mean_parasites_per_vacuole"] == 2.0
    assert s.loc[1, "max_parasites_per_vacuole"] == 2
    assert (
        abs(s.loc[1, "mean_vacuole_ratio_intden"] - vdf.loc[1, "ratio_intden"]) < 1e-9
    )
    # Host 3 has no vacuoles: zero counts, NaN ratios
    assert s.loc[3, "vacuole_area_px_total"] == 0.0
    assert np.isnan(s.loc[3, "mean_vacuole_ratio_intden"])


def test_host_summary_on_empty_vacuole_frame():
    s = host_vacuole_summary(pd.DataFrame(), host_ids=[1, 2])
    assert list(s.index) == [1, 2]
    assert (s["vacuole_area_px_total"] == 0.0).all()


def test_measure_hosts_excludes_whole_vacuole_when_given():
    """Host cytosol must exclude the vacuole lumen, not just parasite bodies."""
    hosts, vacs, paras, img = _scene()
    df = measure_hosts(
        host_labels=hosts,
        para_labels=paras,
        image=img,
        para_to_host=PARA_TO_HOST,
        vac_to_host=VAC_TO_HOST,
        dilation_px=0,
        exclude_labels=vacs,
    )
    # Host 1 cytosol = 324 px total - 64 px vacuole = 260 px, all at cptsa 10
    assert df.loc[1, "area_px"] == 324.0 - 64.0
    assert abs(df.loc[1, "ratio_intden"] - 1.0) < 1e-6
    assert df.loc[1, "excluded_area_px"] == 64.0


def test_measure_hosts_defaults_to_parasite_exclusion():
    """Without exclude_labels the old behaviour (parasite pixels) is kept."""
    hosts, vacs, paras, img = _scene()
    df = measure_hosts(
        host_labels=hosts,
        para_labels=paras,
        image=img,
        para_to_host=PARA_TO_HOST,
        vac_to_host=VAC_TO_HOST,
        dilation_px=0,
    )
    assert df.loc[1, "area_px"] == 324.0 - 18.0  # two 9 px parasites
    assert df.loc[1, "excluded_area_px"] == 18.0


# ── Vacuole counting independent of parasite detection ───────────────────────


def _empty_vacuole_scene():
    """Host 1 holds vacuole 1 (with a parasite) and vacuole 2 (no parasite)."""
    hosts = np.zeros((40, 40), dtype=np.int32)
    hosts[2:38, 2:38] = 1
    vacs = np.zeros_like(hosts)
    vacs[5:13, 5:13] = 1  # 64 px, has a parasite
    vacs[20:28, 20:28] = 2  # 64 px, no parasite detected inside
    paras = np.zeros_like(hosts)
    paras[6:10, 6:10] = 1
    img = np.ones((40, 40, 2), dtype=np.float32)
    return hosts, vacs, paras, img


def test_assign_vacuoles_by_mask_overlap_finds_empty_vacuoles():
    from napari_peredox._host import assign_vacuoles_to_hosts

    hosts, vacs, _paras, _img = _empty_vacuole_scene()
    v2h, dropped = assign_vacuoles_to_hosts(vacs, hosts)
    assert v2h == {1: 1, 2: 1}  # both vacuoles found, not just the occupied one
    assert dropped == []


def test_assign_vacuoles_drops_vacuole_outside_any_host():
    from napari_peredox._host import assign_vacuoles_to_hosts

    hosts = np.zeros((30, 30), dtype=np.int32)
    hosts[2:12, 2:12] = 1
    vacs = np.zeros_like(hosts)
    vacs[20:26, 20:26] = 1  # entirely outside the host
    v2h, dropped = assign_vacuoles_to_hosts(vacs, hosts)
    assert v2h == {}
    assert dropped == [1]


def test_empty_vacuole_counted_and_measured():
    from napari_peredox._host import assign_vacuoles_to_hosts

    hosts, vacs, paras, img = _empty_vacuole_scene()
    v2h, _ = assign_vacuoles_to_hosts(vacs, hosts)
    h = measure_hosts(
        hosts, paras, img, para_to_host={1: 1}, vac_to_host=v2h,
        dilation_px=0, exclude_labels=vacs,
    )
    assert int(h.loc[1, "n_vacuoles"]) == 2  # was 1 before the fix

    vdf = measure_vacuoles_in_hosts(vacs, paras, img, v2h, vacuole_map={1: 1})
    assert sorted(vdf.index.tolist()) == [1, 2]
    assert int(vdf.loc[2, "parasites_per_vacuole"]) == 0

    s = host_vacuole_summary(vdf, host_ids=[1])
    assert s.loc[1, "vacuole_area_px_total"] == 128.0  # both vacuoles
    assert int(s.loc[1, "n_vacuoles_with_parasites"]) == 1
