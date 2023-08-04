import pendulum
import pytest

from qwip import qsettings
from qwip.data.filesystem import date, make_data_directory, timestamp, uuid


def test_date_rule(fixed_time):
    ref_date = fixed_time.date()

    assert date() == ref_date.format("YYYY-MM-DD")
    assert date(date_fmt="YYMMDD") == ref_date.format("YYMMDD")

    with qsettings.context(settings={"data/date_fmt": "DDMMYYYY"}):
        assert date() == ref_date.format("DDMMYYYY")

    assert date("prefix_{date}") == f"prefix_{ref_date}"

    pendulum.set_test_now()


def test_make_data_directory(tmp_path, fixed_time):
    base = str(tmp_path.resolve())

    with qsettings.context(settings={"data/base_directory": base}):
        dir = make_data_directory()

        assert dir.exists()
        assert dir == tmp_path / str(fixed_time.date())
