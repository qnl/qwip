from __future__ import annotations

import sys
import types
from unittest.mock import patch

from qwip_explorer import launcher


def test_launcher_uses_installed_application_and_localhost(monkeypatch):
    calls = []
    cli = types.ModuleType("streamlit.web.cli")
    cli.main = lambda: calls.append(list(sys.argv))
    web = types.ModuleType("streamlit.web")
    web.cli = cli
    streamlit = types.ModuleType("streamlit")
    streamlit.web = web
    previous_argv = list(sys.argv)
    with patch.dict(
        sys.modules,
        {"streamlit": streamlit, "streamlit.web": web, "streamlit.web.cli": cli},
    ):
        launcher.main(["--port", "9123", "--server.headless=true"])
    sys.argv = previous_argv

    assert calls == [[
        "streamlit", "run", str(launcher.application_path()),
        "--server.address=127.0.0.1", "--server.port=9123", "--server.headless=true",
    ]]
