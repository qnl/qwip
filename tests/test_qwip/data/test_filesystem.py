from pathlib import Path

import pandas as pd
import pendulum
import pytest

from qwip import qsettings
from qwip.data.filesystem import (
    DataSaver,
    ParquetDataSaver,
    add_extension,
    camel_to_kebab,
    date,
    make_data_directory,
)
from qwip.processing.data_processor import MeasurementResult
from qwip.processing.processors import (
    GMMClassification,
    IQResult,
    ReadoutBitstring,
    ReadoutHistogram,
    StatePopulations,
)


@pytest.mark.parametrize(
    "name,ext,expected",
    [
        ("file", ".csv", "file.csv"),
        ("file", "csv", "file.csv"),
        ("file.csv", "csv", "file.csv"),
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

    pendulum.set_test_now()


def test_make_data_directory(tmp_path, fixed_time):
    base = str(tmp_path.resolve())

    with qsettings.context(settings={"data/base_directory": base}):
        dir = make_data_directory()

        assert dir.exists()
        assert dir == tmp_path / str(fixed_time.date())


class TestDataSaver:
    @pytest.mark.parametrize(
        "processors",
        [
            tuple(),
            (ReadoutHistogram,),
            (ReadoutBitstring, ReadoutHistogram, StatePopulations),
        ],
    )
    def test_get_result_processor(self, processors):
        result = MeasurementResult(
            name="result",
            data=pd.DataFrame(),
            processors=tuple(p() for p in processors),
        )

        expected = processors[-1] if processors else None
        assert DataSaver.get_result_processor(result) == expected

    def test_get_result_processor_exception(self):
        result = dict(
            R0=MeasurementResult(name="R0", data=pd.DataFrame()),
            R1=MeasurementResult(
                name="R1",
                data=pd.DataFrame(),
                processors=(ReadoutBitstring(), StatePopulations()),
            ),
        )

        with pytest.raises(ValueError):
            DataSaver.get_result_processor(result)

    def test_validate_directory(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            DataSaver(directory=tmp_path / "random")

    def test_split_result(self, fixed_time, rng):
        shape = (21, 2048, 2)

        result = {
            k: IQResult.random(shape, rng=rng, name=k) for k in ("R0", "R1", "R2")
        }

        metadata, data = DataSaver.split_result(result)

        assert metadata == {
            k: {
                "name": k,
                "timestamp": fixed_time.isoformat(),
                "processors": [],
                "__class__": "IQResult",
            }
            for k in result
        }
        assert data.columns.equals(
            pd.MultiIndex.from_tuples(((k, "IQ") for k in result), names=["key", None])
        )


class TestParquetDataSaver:
    @pytest.fixture
    def datasaver(self, tmp_path):
        return ParquetDataSaver(directory=tmp_path)

    def test_get_directory(self, datasaver, tmp_path):
        result_id = "result_id"

        assert datasaver.get_directory(result_id) == tmp_path / result_id
        assert (tmp_path / result_id).exists()

    @pytest.mark.parametrize(
        "processor,filename",
        [
            (GMMClassification, "gmm-classification.parquet"),
            (None, "raw.parquet"),
            (StatePopulations, "state-populations.parquet"),
        ],
    )
    def get_filename(self, processor, filename, datasaver):
        assert datasaver.get_filename(processor) == Path(filename)

    @pytest.mark.parametrize(
        "filename,processor",
        [
            ("gmm-classification", GMMClassification),
            ("raw", None),
            ("state-populations", StatePopulations),
            ("random", ValueError),
        ],
    )
    def test_processor_from_filename(self, filename, processor, datasaver):
        if processor is ValueError:
            with pytest.raises(processor):
                datasaver._processor_from_filename(filename)

        else:
            cache_hits = datasaver._processor_from_filename.cache_info().hits
            assert datasaver._processor_from_filename(filename) is processor
            datasaver._processor_from_filename(filename)
            assert datasaver._processor_from_filename.cache_info().hits > cache_hits

    def test_get_result_types(self, datasaver):
        processors = {None, GMMClassification, StatePopulations}

        folder = datasaver.get_directory("result_id")
        for p in processors:
            (folder / datasaver.get_filename(p)).touch()

        assert datasaver.get_result_types("result_id") == processors
