import itertools as it

import pendulum
import pytest

from qwip import qsettings
from qwip.data.filesystem import (
    add_extension,
    camel_to_kebab,
    date,
    make_data_directory,
)


@pytest.mark.parametrize(
    "name,ext,expected",
    [
        ("file", ".csv", "file.csv"),
        ("file", "csv", "file.csv"),
        ("file.csv", "csv", "file.csv"),
        ("", "", ""),
        ("file.", ".csv", "file.csv"),
        (".gitignore", "", ".gitignore"),
    ],
)
def test_add_extension(name, ext, expected):
    assert add_extension(name, ext) == expected


@pytest.mark.parametrize(
    "name,expected",
    [
        ("CamelCase", "camel-case"),
        ("PopulationResult", "population-result"),
        ("StatePopulations", "state-populations"),
        ("GMMClassification", "gmm-classification"),
    ],
)
def test_camel_to_kebab(name, expected):
    assert camel_to_kebab(name) == expected


def test_date_rule(fixed_time):
    ref_date = fixed_time.date()

    assert date() == ref_date.format("YYYY-MM-DD")
    assert date(date_fmt="YYMMDD") == ref_date.format("YYMMDD")

    with qsettings.context(settings={"data/date_fmt": "DDMMYYYY"}):
        assert date() == ref_date.format("DDMMYYYY")

    assert date("prefix_{date}") == f"prefix_{ref_date}"


def test_make_data_directory(tmp_path, fixed_time):
    base = str(tmp_path.resolve())

    with qsettings.context(settings={"data/base_directory": base}):
        dir = make_data_directory()

        assert dir.exists()
        assert dir == tmp_path / str(fixed_time.date())
