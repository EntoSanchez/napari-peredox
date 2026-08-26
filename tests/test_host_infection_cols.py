import numpy as np
import pandas as pd

from napari_peredox._host import INFECTION_COLS, drop_infection_columns


def test_drops_exactly_the_infection_columns():
    df = pd.DataFrame(
        {
            "ratio_intden": [1.5],
            "area_px": [100.0],
            "infected": [False],
            "n_parasites": [0],
            "n_vacuoles": [0],
            "parasite_area_px": [0.0],
            "cytosol_empty": [False],
            "on_border": [True],
            "host_area_px_total": [100.0],
        }
    )
    out = drop_infection_columns(df)
    assert list(out.columns) == [
        "ratio_intden",
        "area_px",
        "on_border",
        "host_area_px_total",
    ]
    # The original frame is untouched (helper returns a copy)
    assert "infected" in df.columns


def test_infection_cols_constant_matches_measure_hosts_metadata():
    assert INFECTION_COLS == [
        "infected",
        "n_parasites",
        "n_vacuoles",
        "parasite_area_px",
        "cytosol_empty",
    ]


def test_missing_columns_are_tolerated():
    df = pd.DataFrame({"ratio_intden": [np.nan], "n_parasites": [0]})
    out = drop_infection_columns(df)
    assert list(out.columns) == ["ratio_intden"]


def test_empty_frame():
    out = drop_infection_columns(pd.DataFrame())
    assert out.empty
