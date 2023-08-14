import json
import re
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Callable, Literal
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

DIRECTORY_RULES: dict[str, Callable] = FlatDict()
FILENAME_RULES: dict[str, Callable] = FlatDict()

camel2kebab_1 = re.compile(r"(.)([A-Z][a-z]+)")
camel2kebab_2 = re.compile(r"([a-z0-9])([A-Z])")


class DataFormat(str, Enum):
    csv = "csv"
    parquet = "parquet"
    feather = "feather"


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


@qfrozen
class DataSaver:
    directory: Path = field()

    @directory.validator
    def _validate_directory(self, attribute, value):
        if not value.exists():
            raise FileNotFoundError(f"Path '{value}' does not exist!")

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
        """Returns the directory for a given result id.

        The result directory will be a subdirectory of the datasaver directory.

        Args:
            result_id: A result identifier string.
            mkdir: If `True`, creates the directory if it does not exist.

        Returns:
            The result directory.
        """
        folder = self.directory / Path(result_id)
        if mkdir:
            folder.mkdir(exist_ok=True)
        return folder

    def get_filename(
        self,
        processor: type[DataProcessor] | None = None,
        fmt: DataFormat | Literal[".csv", ".parquet", ".feather"] = DataFormat.parquet,
    ) -> Path:
        """Returns the filename for a given processor type.

        The filename will be a kebab-case variant of the processor class name with
        the proper file extension.

        Args:
            processor: A processor class.
            fmt: An allowed data format.

        Returns:
            A relative file path for the given processor type and file format.
        """
        if processor is None:
            pname = "raw"
        else:
            pname = camel_to_kebab(re.sub(r"[\[\]]", "", generic_to_string(processor)))

        return Path(add_extension(pname, fmt))

    def get_save_path(
        self,
        result_id: str,
        result: MeasurementResult | dict[str, MeasurementResult],
        overwrite: bool = False,
    ) -> Path:
        """Returns the absolute save path for a given result and result id.

        Args:
            result id: The result identifier.
            result: The measurement result that is being saved.
            overwrite: Whether to raise an error if the save path already exists.

        Returns:
            An absolute path for a given result id and result.
        """
        processor = type(self).get_result_processor(result)
        folder = self.get_directory(result_id)
        filename = self.get_filename(processor)
        outpath = folder / filename

        if outpath.exists() and not overwrite:
            raise FileExistsError(
                f"File {outpath.resolve()} already exists. Pass `overwrite = True` to overwrite an existing file."
            )

        return outpath.resolve()

    def result_types(
        self,
        result_id: str,
        fmt: DataFormat
        | Literal[".csv", ".parquet", ".feather"]
        | None = DataFormat.parquet,
    ) -> set[type[DataProcessor]]:
        """Returns the set of saved result types.

        Args:
            result_id: The result identifier.
            fmt: The data format to look for.

        Returns:
            A set containing the result types that were saved.
        """
        folder = self.get_directory(result_id, mkdir=False)

        try:
            return {
                self._processor_from_filename(f)
                for f in folder.iterdir()
                if (fmt is None) or (f.suffix[1:] == fmt)
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

    def save(
        self,
        result_id: str,
        result: MeasurementResult | dict[str, MeasurementResult],
        fmt: DataFormat | Literal[".csv", ".parquet", ".feather"] = DataFormat.parquet,
        overwrite: bool = False,
    ) -> Path:
        """Saves a measurement result.

        Args:
            result_id: An identifier for the measurement result.
            result: A dictionary of measurement results or a single measurement result.
            fmt: The data format used to save the results.
            overwrite: If `True`, will overwrite any existing data with the same
                filename.

        Returns:
            The full path to the data file.
        """
        if not isinstance(fmt, DataFormat):
            fmt = DataFormat[fmt]

        return getattr(self, f"save_{fmt}")(result_id, result, overwrite)

    def load(self, filename: Path) -> dict[str, MeasurementResult]:
        """Loads a measurement result.

        Args:
            filename: The path to the data file. If a relative path is passed, it is
                assumed to be referenced to the `directory` attribute of the datasaver.

        Returns:
            The reloaded measurement result.
        """
        fmt = DataFormat[filename.suffix[1:]]

        if not filename.is_absolute():
            filename = self.directory / filename

        return getattr(self, f"load_{fmt}")(filename)

    def save_csv(
        self,
        result_id: str,
        result: MeasurementResult | dict[str, MeasurementResult],
        overwrite: bool = False,
    ) -> Path:
        """Saves a measurement result as a csv.

        Measurement metadata is saved as a json file with the same filename as the
        result data.

        Args:
            result_id: An identifier for the measurement result.
            result: A dictionary of measurement results or a single measurement result.
            overwrite: If `True`, will overwrite any existing data with the same
                filename.

        Returns:
            The full path to the data file.
        """
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

    def load_csv(self, filename: Path) -> dict[str, MeasurementResult]:
        """Loads a measurement result from a csv.

        The json metadata for the measurement result must also be present in the same
        directory with the same filename but suffix  replaced by `.json`.

        Args:
            filename: The path to the data file. If a relative path is passed, it is
                assumed to be referenced to the `directory` attribute of the datasaver.

        Returns:
            The reloaded measurement result.
        """
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

    def save_parquet(
        self,
        result_id: str,
        result: MeasurementResult | dict[str, MeasurementResult],
        overwrite: bool = False,
    ) -> Path:
        """Saves a measurement result as a parquet file.

        Complex results are converted to two real columns prior to saving because Arrow
        has no built in support for complex data types.

        Args:
            result_id: An identifier for the measurement result.
            result: A dictionary of measurement results or a single measurement result.
            overwrite: If `True`, will overwrite any existing data with the same
                filename.

        Returns:
            The full path to the data file.
        """
        savepath = self.get_save_path(result_id, result, overwrite=overwrite)

        metadata, data = type(self).split_result(result)
        table = to_arrow_table(metadata, data)

        pq.write_table(table, savepath)

        return savepath

    def load_parquet(self, filename: Path) -> dict[str, MeasurementResult]:
        """Loads a measurement result from a parquet file.

        Any complex data columns that were expanded to two real columns for saving will
        be converted back to a complex data column.

        Args:
            filename: The path to the data file. If a relative path is passed, it is
                assumed to be referenced to the `directory` attribute of the datasaver.

        Returns:
            The reloaded measurement result.
        """
        table = pq.read_table(filename)
        result, data = from_arrow_table(table)

        for k, df in data.groupby(level="key", axis="columns"):
            result[k]["data"] = df.droplevel("key", axis="columns")

        return qwip.converter.structure(result, dict[str, MeasurementResult])

    def save_feather(
        self,
        result_id: str,
        result: MeasurementResult | dict[str, MeasurementResult],
        overwrite: bool = False,
    ) -> Path:
        """Saves a measurement result as a feather file.

        Complex results are converted to two real columns prior to saving because Arrow
        has no built in support for complex data types.

        Args:
            result_id: An identifier for the measurement result.
            result: A dictionary of measurement results or a single measurement result.
            overwrite: If `True`, will overwrite any existing data with the same
                filename.

        Returns:
            The full path to the data file.
        """
        savepath = self.get_save_path(result_id, result, overwrite=overwrite)

        metadata, data = type(self).split_result(result)
        table = to_arrow_table(metadata, data)

        pf.write_feather(table, savepath)

        return savepath

    def load_feather(self, filename: Path) -> dict[str, MeasurementResult]:
        """Loads a measurement result from a feather file.

        Any complex data columns that were expanded to two real columns for saving will
        be converted back to a complex data column.

        Args:
            filename: The path to the data file. If a relative path is passed, it is
                assumed to be referenced to the `directory` attribute of the datasaver.

        Returns:
            The reloaded measurement result.
        """
        table = pf.read_table(filename)
        result, data = from_arrow_table(table)

        for k, df in data.groupby(level="key", axis="columns"):
            result[k]["data"] = df.droplevel("key", axis="columns")

        return qwip.converter.structure(result, dict[str, MeasurementResult])
