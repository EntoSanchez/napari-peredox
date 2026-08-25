import pandas as pd

from napari_peredox._io import append_curated_annotations


def test_default_csv_name_unchanged(tmp_path):
    path = append_curated_annotations(
        decisions={1: 1},
        features=pd.DataFrame(),
        image_stem="img",
        annotations_dir=tmp_path,
    )
    assert path.name == "curated_features.csv"
    assert path.exists()


def test_custom_csv_name_writes_host_file(tmp_path):
    path = append_curated_annotations(
        decisions={1: 1, 2: 0},
        features=pd.DataFrame(),
        image_stem="img",
        annotations_dir=tmp_path,
        csv_name="curated_host_features.csv",
    )
    assert path.name == "curated_host_features.csv"
    df = pd.read_csv(path)
    assert len(df) == 2
    assert not (tmp_path / "curated_features.csv").exists()


def test_load_classifier_custom_filename(tmp_path):
    from napari_peredox._learning import load_classifier

    # No model files exist -> both return None, but neither must raise
    assert load_classifier(tmp_path) is None
    assert load_classifier(tmp_path, filename="curated_host_features.joblib") is None
