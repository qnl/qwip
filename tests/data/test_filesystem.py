import pendulum
import pytest

from qwip import qsettings
from qwip.data.filesystem import date, make_data_directory, timestamp, uuid


@pytest.fixture
def pendulum_datetime():
    return pendulum.datetime(2015, 3, 14, 9, 26, 53, tz="America/Los_Angeles")


def test_date_rule(pendulum_datetime):
    pendulum.set_test_now(pendulum_datetime)
    ref_date = pendulum_datetime.date()

    assert date() == ref_date.format("YYYY-MM-DD")
    assert date(date_fmt="YYMMDD") == ref_date.format("YYMMDD")

    with qsettings.context(settings={"data/date_fmt": "DDMMYYYY"}):
        assert date() == ref_date.format("DDMMYYYY")

    assert date("prefix_{date}") == f"prefix_{ref_date}"

    pendulum.set_test_now()


def test_make_data_directory(tmp_path, pendulum_datetime):
    pendulum.set_test_now(pendulum_datetime)

    base = str(tmp_path.resolve())

    with qsettings.context(settings={"data/base_directory": base}):
        dir = make_data_directory()

        assert dir.exists()
        assert dir == tmp_path / str(pendulum_datetime.date())

    pendulum.set_test_now(pendulum_datetime)
