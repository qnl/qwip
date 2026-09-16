"""Local launcher for the optional Streamlit Explorer."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def application_path() -> Path:
    """Return the installed Streamlit application path, independent of cwd."""
    return Path(__file__).with_name("app.py")


def main(argv: list[str] | None = None) -> None:
    """Start the Explorer bound to localhost on a configurable port."""
    parser = argparse.ArgumentParser(description="Launch the QWIP Datastore Explorer.")
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("QWIP_EXPLORER_PORT", "8501")),
        help="Localhost port (default: QWIP_EXPLORER_PORT or 8501).",
    )
    args, streamlit_args = parser.parse_known_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    try:
        from streamlit.web import cli as stcli
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "The optional Explorer is not installed. Run: pip install 'qwip[explorer]'"
        ) from exc

    sys.argv = [
        "streamlit",
        "run",
        str(application_path()),
        "--server.address=127.0.0.1",
        f"--server.port={args.port}",
        *streamlit_args,
    ]
    stcli.main()
