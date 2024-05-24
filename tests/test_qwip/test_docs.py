import runpy
from pathlib import Path

import pytest

# Inspired by https://stackoverflow.com/a/56813896

doc_examples_path = Path(__file__, "../../../examples/docs").resolve()
scripts = doc_examples_path.glob("*.py")


def name_from_path(path: Path) -> str:
    return path.name


@pytest.mark.parametrize("script", scripts, ids=name_from_path)
def test_documentation_examples(script):
    runpy.run_path(script)
