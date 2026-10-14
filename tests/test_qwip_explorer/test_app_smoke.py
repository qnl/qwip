from pathlib import Path

import pytest


def test_app_renders_connection_form_and_switches_storage_mode(monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("QWIP_STORAGE_URL", "http://assets.example:4002")
    monkeypatch.setenv("QWIP_LOCAL_DATASTORE", "C:/data/local_datastore")
    app = Path(__file__).parents[2] / "src" / "qwip_explorer" / "app.py"
    at = AppTest.from_file(app).run()

    assert not at.exception
    assert at.title[0].value == "QWIP Datastore Explorer"
    assert at.text_input[0].value == ""
    assert at.text_input[-1].value == "http://assets.example:4002"

    at.radio[0].set_value("Local folder").run()
    assert not at.exception
    assert at.text_input[-1].value == "C:/data/local_datastore"
