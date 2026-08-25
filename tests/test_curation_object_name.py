"""Signature-level test: _curation imports Qt, so avoid instantiating widgets."""

import inspect


def test_vacuole_curation_widget_accepts_object_name():
    from napari_peredox._curation import VacuoleCurationWidget

    sig = inspect.signature(VacuoleCurationWidget.__init__)
    assert "object_name" in sig.parameters
    assert sig.parameters["object_name"].default == "vacuole"
