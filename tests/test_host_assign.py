import numpy as np

from napari_peredox._host import assign_to_hosts


def _canvas():
    """20x20 image: host 1 = left half, host 2 = right half, 2-col background gap."""
    hosts = np.zeros((20, 20), dtype=np.int32)
    hosts[:, 0:9] = 1
    hosts[:, 11:20] = 2
    return hosts


def test_clean_containment():
    hosts = _canvas()
    paras = np.zeros_like(hosts)
    paras[2:5, 2:5] = 1  # fully inside host 1
    p2h, v2h, dropped = assign_to_hosts(paras, hosts)
    assert p2h == {1: 1}
    assert dropped == []


def test_straddling_parasite_goes_to_majority_host():
    hosts = _canvas()
    paras = np.zeros_like(hosts)
    paras[5, 6:14] = (
        1  # cols 6-8 in host1 (3 px), 11-13 in host2 (3 px), 9-10 bg (2 px)
    )
    paras[6, 6:9] = 1  # 3 more px in host 1 -> host 1 majority
    p2h, _, dropped = assign_to_hosts(paras, hosts)
    assert p2h == {1: 1}
    assert dropped == []


def test_tie_resolves_to_lowest_host_id():
    hosts = _canvas()
    paras = np.zeros_like(hosts)
    paras[5, 7:9] = 1  # 2 px in host 1
    paras[5, 11:13] = 1  # 2 px in host 2
    p2h, _, _ = assign_to_hosts(paras, hosts)
    assert p2h == {1: 1}


def test_background_majority_is_dropped():
    hosts = _canvas()
    paras = np.zeros_like(hosts)
    paras[5, 8:12] = 1  # 1 px host1, 2 px background (cols 9,10), 1 px host2
    p2h, _, dropped = assign_to_hosts(paras, hosts)
    assert p2h == {}
    assert dropped == [1]


def test_vacuole_assignment_follows_member_pixel_majority():
    hosts = _canvas()
    paras = np.zeros_like(hosts)
    paras[2:4, 2:4] = 1  # 4 px, host 1
    paras[2:4, 5:7] = 2  # 4 px, host 1
    paras[10:16, 12:18] = 3  # 36 px, host 2
    vacuole_map = {1: 10, 2: 10, 3: 20}
    p2h, v2h, _ = assign_to_hosts(paras, hosts, vacuole_map)
    assert p2h == {1: 1, 2: 1, 3: 2}
    assert v2h == {10: 1, 20: 2}


def test_no_vacuole_map_returns_empty_vac_dict():
    hosts = _canvas()
    paras = np.zeros_like(hosts)
    paras[2:5, 2:5] = 1
    _, v2h, _ = assign_to_hosts(paras, hosts)
    assert v2h == {}
