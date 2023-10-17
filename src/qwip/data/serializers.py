import json
import re
from abc import ABCMeta, abstractproperty
from functools import lru_cache
from io import BufferedReader
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as pf
import pyarrow.parquet as pq

import qwip
from qwip.attrs import qfrozen
from qwip.data.filesystem import add_extension, camel_to_kebab
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

SERIALIZERS = dict()


@qfrozen
class Serializer(metaclass=ABCMeta):
    @abstractproperty
    def formats(self) -> frozenset[str]:
        ...

    def to_stream(
        self, obj: Any, *, fmt: str | None = None, **kwargs
    ) -> BufferedReader:
        if fmt not in self.formats():
            raise ValueError(f"{fmt} is not a known format for {type(self).__name__}")

        return getattr(self, f"to_stream_{fmt}")(obj, **kwargs)

    def from_stream(
        self, stream: BufferedReader, *, fmt: str | None = None, **kwargs
    ) -> Any:
        if fmt not in self.formats():
            raise ValueError(f"{fmt} is not a known format for {type(self).__name__}")

        return getattr(self, f"to_stream_{fmt}")(stream, **kwargs)


def to_arrow_table(metadata: dict, data: pd.DataFrame) -> pa.Table:
    """Converts a metadata dictionary and dataframe to an arrow table.

    Args:
        metadata: Metadata to add. This must be a json serializable dictionary.
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
        table.schema.metadata
        | dict(qwip=metadata, qwip_complex=str(has_complex).encode())
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
    if table.schema.metadata[b"qwip_complex"] == b"True":
        data = dataframe_real_to_complex(data)

    return metadata, data


@qfrozen
class DataFrameSerializer(Serializer):
    @property
    def formats(self) -> frozenset[str]:
        return frozenset("parquet", "feather", "csv")

    def to_stream_parquet(
        self,
        data: pd.DataFrame,
        metadata: dict = {},
    ) -> BufferedReader:
        table = to_arrow_table(metadata, data)

        with pa.BufferOutputStream() as stream:
            pq.write_table(table, stream)

        return pa.input_stream(stream.getvalue())

    def from_stream_parquet(self, stream: BufferedReader) -> tuple[pd.DataFrame, dict]:
        table = pq.read_table(stream)

        try:
            stream.close()
        except AttributeError:
            ...

        metadata, data = from_arrow_table(table)
        return data, metadata

    def to_stream_feather(
        self, data: pd.DataFrame, metadata: dict = {}
    ) -> BufferedReader:
        table = to_arrow_table(metadata, data)

        with pa.BufferOutputStream() as stream:
            pf.write_feather(table, stream)

        return pa.input_stream(stream.getvalue())

    def from_stream_feather(self, stream: BufferedReader) -> tuple[pd.DataFrame, dict]:
        table = pf.read_table(stream)

        try:
            stream.close()
        except AttributeError:
            ...

        metadata, data = from_arrow_table(table)
        return data, metadata


@qfrozen
class ResultSerializer(DataFrameSerializer):
    def to_stream_parquet(
        self, result: MeasurementResult | dict[str, MeasurementResult]
    ) -> BufferedReader:
        metadata, data = type(self).split_result(result)

        return super().to_stream_parquet(data, metadata)

    def from_stream_parquet(
        self, stream: BufferedReader
    ) -> dict[str, MeasurementResult]:
        data, metadata = super().from_stream_parquet(stream)

        for k, df in data.groupby(level="key", axis="columns"):
            metadata[k]["data"] = df.droplevel("key", axis="columns")

        return qwip.converter.structure(metadata, dict[str, MeasurementResult])

    def to_stream_feather(
        self, result: MeasurementResult | dict[str, MeasurementResult]
    ) -> BufferedReader:
        metadata, data = type(self).split_result(result)

        return super().to_stream_feather(data, metadata)

    def from_stream_feather(
        self, stream: BufferedReader
    ) -> dict[str, MeasurementResult]:
        data, metadata = super().from_stream_feather(stream)

        for k, df in data.groupby(level="key", axis="columns"):
            metadata[k]["data"] = df.droplevel("key", axis="columns")

        return qwip.converter.structure(metadata, dict[str, MeasurementResult])

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

    def get_filename(
        self,
        processor: type[DataProcessor] | None = None,
        fmt: str = "parquet",
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

    @lru_cache(maxsize=32)
    def processor_from_filename(self, filename: str | Path) -> type[DataProcessor]:
        if isinstance(filename, str):
            filename = Path(filename)

        for processor in (None, *DATA_PROCESSORS.registered.values()):
            if self.get_filename(processor).stem == filename.stem:
                return processor

        raise ValueError(f"{filename} does not correspond to a known processor.")
