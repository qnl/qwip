import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict
from uuid import uuid4

import numpy as np
import pandas as pd
import pendulum
import pyarrow as pa
import pyarrow.feather as pf
import pyarrow.parquet as pq
from attrs import field
from loguru import logger
from numpy.core.numeric import complexfloating

import qwip
from qwip.attrs import qdefine, qfrozen
from qwip.defaults import dynamic_default
from qwip.flatdict import FlatDict
from qwip.processing.data_processor import (
    DATA_PROCESSORS,
    DataProcessor,
    MeasurementResult,
)
from qwip.processing.processors import (
    dataframe_complex_to_real,
    dataframe_real_to_complex,
)
from qwip.typing import generic_to_string

DIRECTORY_RULES: Dict[str, Callable] = FlatDict()
FILENAME_RULES: Dict[str, Callable] = FlatDict()

camel2kebab_1 = re.compile(r"(.)([A-Z][a-z]+)")
camel2kebab_2 = re.compile(r"([a-z0-9])([A-Z])")


def add_extension(name: str, ext: str) -> str:
    """Add a file extension to a filename.

    If the filename already ends with the specified extension, no changes are made.

    Args:
        name: A filename.
        ext: An extension string. An extension can be specified with or without the '.'.
            For example, `.csv` or `csv` are both valid and will yield the same result.

    Returns:
        A modified filename with the specified extension.
    """
    ext = ext if ext.startswith(".") else f".{ext}"
    name = name if name.endswith(ext) else f"{name}{ext}"

    return name


def camel_to_kebab(name: str) -> str:
    """Converts a `CamelCase` string to a `kebab-case` string.

    See [StackOverflow post](https://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-snake-case)
    for reference.

    Args:
        name: The string to convert.

    Returns:
        The converted string.
    """
    name = camel2kebab_1.sub(r"\1-\2", name)
    return camel2kebab_2.sub(r"\1-\2", name).lower()


@dynamic_default(base="data/base_directory", rule="data/directory_rule")
def get_data_directory(base: str = None, dirname: str = "", rule: str = None, **kwargs):
    """ """
    if not dirname:
        dirname = DIRECTORY_RULES[rule](**kwargs)

    base = Path(base).resolve()
    directory = base / dirname

    return directory


@dynamic_default(
    base="data/base_directory",
    rule="data/directory_rule",
    exist_ok="data/directory_exist_ok",
)
def make_data_directory(
    base: str = None,
    dirname: str = "",
    rule: str = None,
    exist_ok: bool = None,
    **kwargs,
):
    directory = get_data_directory(base, dirname, rule, **kwargs)
    directory.mkdir(parents=True, exist_ok=exist_ok)

    return directory


def get_rules():
    return list(DIRECTORY_RULES.keys())


def directory_rule(func):
    DIRECTORY_RULES[func.__name__] = func
    return func


@directory_rule
@dynamic_default(date_fmt="data/date_fmt")
def date(name_fmt: str = "{date}", date_fmt: str = None):
    date = pendulum.now().format(date_fmt)
    return name_fmt.format(date=date)


@directory_rule
def uuid(name_fmt: str = "{uuid}"):
    return name_fmt.format(uuid=uuid4())


@directory_rule
def timestamp(name_fmt: str = "{timestamp}"):
    return name_fmt.format(timestamp=pendulum.now().int_timestamp)


@qfrozen
class DataSaver:
    directory: Path = field()

    @directory.validator
    def _validate_directory(self, attribute, value):
        if not value.exists():
            raise FileNotFoundError(f"Path '{value}' does not exist!")

    @property
    def extension(self) -> str:
        return ""

    @classmethod
    def get_result_processor(
        cls, result: MeasurementResult | dict[str, MeasurementResult]
    ) -> type[DataProcessor]:
        """Get the common final processor from all results in the result dictionary.

        Args:
            result: A measurement result or dictionary of measurement results.

        Returns:
            The common final processor for the result(s).

        Raises:
            ValueError: If the result dictionary contains more than one final processor.
        """
        match result:
            case MeasurementResult():
                return result.final_processor()

        processors = set(r.final_processor() for r in result.values())

        if len(processors) > 1:
            raise ValueError(
                f"Result dictionary must have same final processor. Got {processors}"
            )

        return processors.pop()

    @classmethod
    def split_result(
        cls, result: MeasurementResult | dict[str, MeasurementResult]
    ) -> tuple[dict, pd.DataFrame]:
        """Splits the result dictionary into a metadata dictionary and a dataframe.

        The data contained in each result is concatenated into a single dataframe with
        the outermost columns indexed by measurement key.

        Args:
            result: A single measurement result or dictionary of measurement results.

        Returns:
            A tuple `(metadata, dataframe)`.
        """
        match result:
            case MeasurementResult(name=name):
                result = {name: result}

        metadata = qwip.converter.unstructure(result)
        data = pd.concat(
            [r.data for r in result.values()],
            axis="columns",
            keys=result.keys(),
            names=["key"],
        )

        return metadata, data

    def get_directory(self, result_id: str, mkdir: bool = True) -> Path:
        folder = self.directory / Path(result_id)
        if mkdir:
            folder.mkdir(exist_ok=True)
        return folder

    def get_filename(self, processor: type[DataProcessor] | None = None) -> Path:
        if processor is None:
            pname = "raw"
        else:
            pname = camel_to_kebab(re.sub(r"[\[\]]", "", generic_to_string(processor)))

        return Path(add_extension(pname, self.extension))

    def get_save_path(
        self,
        result_id: str,
        result: MeasurementResult | dict[str, MeasurementResult],
        overwrite: bool = False,
    ) -> Path:
        processor = type(self).get_result_processor(result)
        folder = self.get_directory(result_id)
        filename = self.get_filename(processor)
        outpath = folder / filename

        if outpath.exists() and not overwrite:
            raise FileExistsError(
                f"File {outpath.resolve()} already exists. Pass `overwrite = True` to overwrite an existing file."
            )

        return outpath.resolve()

    def result_types(self, result_id: str) -> set[type[DataProcessor]]:
        folder = self.get_directory(result_id, mkdir=False)

        try:
            return {
                self._processor_from_filename(f)
                for f in folder.iterdir()
                if f.suffix == self.extension
            }
        except FileNotFoundError:
            return set()

    @lru_cache(maxsize=32)
    def _processor_from_filename(self, filename: str | Path) -> type[DataProcessor]:
        if isinstance(filename, str):
            filename = Path(filename)

        for processor in (None, *DATA_PROCESSORS.registered.values()):
            if self.get_filename(processor).stem == filename.stem:
                return processor

        raise ValueError(f"{filename} does not correspond to a known processor.")


class CSVDataSaver(DataSaver):
    @property
    def extension(self) -> str:
        return ".csv"

    def save_data(
        self,
        result_id: str,
        result: MeasurementResult | dict[str, MeasurementResult],
        overwrite: bool = False,
    ) -> Path:
        savepath = self.get_save_path(result_id, result, overwrite=overwrite)
        metadata_savepath = savepath.with_suffix(".json")

        metadata, data = type(self).split_result(result)
        pandas_metadata = dict(
            index=data.index.nlevels,
            header=data.columns.nlevels,
            dtypes=qwip.converter.unstructure(
                [(k, dtype.type) for k, dtype in data.dtypes.items()]
            ),
        )
        with open(metadata_savepath, "w") as f:
            json.dump(dict(qwip=metadata, pandas=pandas_metadata), f)

        data.to_csv(savepath)
        return savepath

    def load_data(self, filename: Path) -> dict[str, MeasurementResult]:
        if not isinstance(filename, Path):
            filename = Path(filename)

        metadata_filename = filename.with_suffix(".json")

        with open(metadata_filename, "r") as f:
            metadata = json.load(f)

        index = list(range(metadata["pandas"]["index"]))
        header = list(range(metadata["pandas"]["header"]))
        dtypes = dict(
            qwip.converter.structure(
                metadata["pandas"]["dtypes"], list[tuple[tuple, type]]
            )
        )

        # Needed for handling complex columns
        replace = dict()
        for k in dtypes:
            if issubclass(dtypes[k], (complex, complexfloating)):
                replace[k] = dtypes[k]
                dtypes[k] = str

        data = pd.read_csv(
            filename, header=header, index_col=index, dtype=dtypes
        ).astype(replace)

        result = metadata["qwip"]
        for k, df in data.groupby(level="key", axis="columns"):
            result[k]["data"] = df.droplevel("key", axis="columns")

        return qwip.converter.structure(result, dict[str, MeasurementResult])


def to_arrow_table(metadata: dict, data: pd.DataFrame) -> pa.Table:
    """Converts a metadata dictionary and dataframe to an arrow table.

    Args:
        metadata: A dictionary containing the result metadata.
        data: A pandas dataframe containing the combined result data.

    Returns:
        A pyarrow table with both the data and metadata.
    """
    metadata = json.dumps(metadata).encode()

    has_complex = data.dtypes.apply(
        lambda d: issubclass(d.type, (np.complexfloating, complex))
    ).any()

    if has_complex:
        data = dataframe_complex_to_real(data, names=("I", "Q"))

    table = pa.Table.from_pandas(data)
    table = table.replace_schema_metadata(
        table.schema.metadata | dict(qwip=metadata, complex=str(has_complex).encode())
    )

    return table


def from_arrow_table(table: pa.Table) -> tuple[dict, pd.DataFrame]:
    """Converts a arrow table to a QWiP metadata dictionary and pandas dataframe.

    Args:
        table: The pyarrow table.

    Returns:
        A tuple `(metadata, data)` from the pyarrow table.
    """
    metadata = json.loads(table.schema.metadata[b"qwip"])

    data = table.to_pandas()
    if table.schema.metadata[b"complex"] == b"True":
        data = dataframe_real_to_complex(data)

    return metadata, data


class ParquetDataSaver(DataSaver):
    @property
    def extension(self) -> str:
        return ".parquet"

    def save_data(
        self,
        result_id: str,
        result: MeasurementResult | dict[str, MeasurementResult],
        overwrite: bool = False,
    ) -> Path:
        savepath = self.get_save_path(result_id, result, overwrite=overwrite)

        metadata, data = type(self).split_result(result)
        table = to_arrow_table(metadata, data)

        pq.write_table(table, savepath)

        return savepath

    def load_data(self, filename: Path) -> dict[str, MeasurementResult]:
        table = pq.read_table(filename)
        result, data = from_arrow_table(table)

        for k, df in data.groupby(level="key", axis="columns"):
            result[k]["data"] = df.droplevel("key", axis="columns")

        return qwip.converter.structure(result, dict[str, MeasurementResult])


class FeatherDataSaver(DataSaver):
    @property
    def extension(self) -> str:
        return ".feather"

    def save_data(
        self,
        result_id: str,
        result: MeasurementResult | dict[str, MeasurementResult],
        overwrite: bool = False,
    ) -> Path:
        savepath = self.get_save_path(result_id, result, overwrite=overwrite)

        metadata, data = type(self).split_result(result)
        table = to_arrow_table(metadata, data)

        pf.write_feather(table, savepath)

        return savepath

    def load_data(self, filename: Path) -> dict[str, MeasurementResult]:
        table = pf.read_table(filename)
        result, data = from_arrow_table(table)

        for k, df in data.groupby(level="key", axis="columns"):
            result[k]["data"] = df.droplevel("key", axis="columns")

        return qwip.converter.structure(result, dict[str, MeasurementResult])
