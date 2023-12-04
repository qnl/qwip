import codecs
import json
import re
from abc import ABCMeta, abstractproperty
from functools import lru_cache, singledispatch
from io import BufferedReader, BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as pf
import pyarrow.parquet as pq
from loguru import logger
from matplotlib.figure import Figure

import qwip
from qwip.attrs import qfrozen
from qwip.data.filesystem import camel_to_kebab
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

try:
    import trueq as tq
except Exception as e:
    logger.exception("Unable to import True-Q", exception=e)

SERIALIZERS: dict[str, "Serializer"] = dict()


@qfrozen
class Serializer(metaclass=ABCMeta):
    @abstractproperty
    def formats(self) -> tuple[str, ...]:
        ...

    @property
    def default_format(self) -> str | None:
        try:
            return self.formats[0]
        except IndexError:
            return None

    @property
    def key(self) -> str:
        return type(self).__name__.lower().replace("serializer", "")

    def to_stream(
        self, obj: Any, *, fmt: str | None = None, **kwargs
    ) -> BufferedReader:
        if fmt is None:
            fmt = self.default_format

        if fmt not in self.formats:
            raise ValueError(f"{fmt} is not a known format for {type(self).__name__}")

        return getattr(self, f"to_stream_{fmt}")(obj, **kwargs)

    def from_stream(
        self, stream: BufferedReader, *, fmt: str | None = None, **kwargs
    ) -> Any:
        if fmt is None:
            fmt = self.default_format

        if fmt not in self.formats:
            raise ValueError(f"{fmt} is not a known format for {type(self).__name__}")

        try:
            obj = getattr(self, f"from_stream_{fmt}")(stream, **kwargs)
        finally:
            try:
                stream.close()
            except Exception as e:
                logger.exception(exception=e)

        return obj

    def get_name(self, obj: Any) -> str:
        """Returns an auto-generated name for the object based on the type."""
        return camel_to_kebab(type(obj).__name__)


@qfrozen
class DefaultSerializer(Serializer):
    @property
    def formats(self) -> tuple[str, ...]:
        return ("json",)

    def to_stream_json(self, obj: Any) -> BufferedReader:
        stream = codecs.getwriter("utf-8")(BytesIO())
        json.dump(qwip.converter.unstructure(obj), stream)
        stream.seek(0)
        return stream

    def from_stream_json(self, stream: BufferedReader, cls: type | None = None) -> Any:
        unstructured = json.load(stream)

        if cls:
            return qwip.converter.structure(unstructured, cls)

        return unstructured


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
    def formats(self) -> tuple[str, ...]:
        return ("parquet", "feather", "csv")

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
        metadata, data = from_arrow_table(table)

        if metadata:
            return data, metadata

        return data

    def to_stream_feather(
        self, data: pd.DataFrame, metadata: dict = {}
    ) -> BufferedReader:
        table = to_arrow_table(metadata, data)

        with pa.BufferOutputStream() as stream:
            pf.write_feather(table, stream)

        return pa.input_stream(stream.getvalue())

    def from_stream_feather(self, stream: BufferedReader) -> tuple[pd.DataFrame, dict]:
        table = pf.read_table(stream)
        metadata, data = from_arrow_table(table)
        if metadata:
            return data, metadata

        return data


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

    def get_name(self, result: MeasurementResult | dict[str, MeasurementResult]) -> str:
        """Returns an auto-generated name from a result.

        Args:
            result: A measurement result.

        Returns:
            A kebab-case name based on the final processor for the result.
        """
        return self.name_from_processor(type(self).get_result_processor(result))

    def name_from_processor(
        self,
        processor: type[DataProcessor] | None = None,
    ) -> Path:
        """Returns the name for a given processor type.

        The name will be a kebab-case variant of the processor class name.

        Args:
            processor: A processor class.

        Returns:
            A name for the given processor type.
        """
        if processor is None:
            return "raw"

        return camel_to_kebab(re.sub(r"[\[\]]", "", generic_to_string(processor)))

    @lru_cache(maxsize=32)
    def processor_from_name(self, name: str) -> type[DataProcessor]:
        for processor in (None, *DATA_PROCESSORS.registered.values()):
            if self.name_from_processor(processor) == name:
                return processor

        raise ValueError(f"{name} does not correspond to a known processor.")


@qfrozen
class MatplotlibSerializer(Serializer):
    """A serializer for matplotlib figures."""

    @property
    def formats(self) -> tuple[str, ...]:
        return ("png", "svg", "pdf")

    def to_stream_png(
        self,
        figure: Figure,
        **kwargs,
    ) -> BufferedReader:
        stream = BytesIO()

        kwargs["format"] = "png"
        figure.savefig(stream, **kwargs)
        stream.seek(0)
        return stream

    def from_stream_png(self, stream: BufferedReader):
        try:
            from IPython.display import Image

            return Image(stream.read())
        except ModuleNotFoundError:
            image = stream.read()
            return image

    def to_stream_svg(
        self,
        figure: Figure,
        **kwargs,
    ) -> BufferedReader:
        stream = BytesIO()

        kwargs["format"] = "svg"
        figure.savefig(stream, **kwargs)
        stream.seek(0)
        return stream

    def from_stream_svg(self, stream: BufferedReader):
        data = stream.read().decode("utf-8")

        try:
            from IPython.display import HTML

            return HTML(data)
        except ModuleNotFoundError:
            return data

    def to_stream_pdf(
        self,
        figure: Figure,
        **kwargs,
    ) -> BufferedReader:
        stream = BytesIO()

        kwargs["format"] = "pdf"
        figure.savefig(stream, **kwargs)
        stream.seek(0)
        return stream

    def from_stream_pdf(self, stream: BufferedReader):
        return stream.read()


@qfrozen
class TrueQSerializer(Serializer):
    """A serializer for TrueQ circuits and circuit collections."""

    @property
    def formats(self) -> tuple[str, ...]:
        return ("tq",)

    def to_stream_tq(
        self,
        circuits,
        **kwargs,
    ):
        stream = BytesIO()
        tq.utils.save(circuits, stream, **kwargs)
        stream.seek(0)
        return stream

    def from_stream_tq(self, stream, **kwargs):
        with NamedTemporaryFile(mode="w+b", delete=False) as f:
            filename = f.name
            f.write(stream.read())
            f.seek(0)

        circuits = tq.utils.load(filename)
        Path(filename).unlink()

        return circuits


def register_serializer(serializer: Serializer) -> str:
    SERIALIZERS[serializer.key] = serializer
    return serializer.key


@singledispatch
def detect_serializer(obj: Any):
    return SERIALIZERS["default"]


@detect_serializer.register(dict)
def detect_dict(obj: dict):
    values = set(type(v) for v in obj.values())

    if len(values) == 1 and issubclass(values.pop(), MeasurementResult):
        return SERIALIZERS["result"]

    return SERIALIZERS["default"]


@detect_serializer.register(MeasurementResult)
def detect_measurement_result(obj: MeasurementResult):
    return SERIALIZERS["result"]


@detect_serializer.register(pd.DataFrame)
def detect_dataframe(obj: pd.DataFrame):
    return SERIALIZERS["dataframe"]


@detect_serializer.register(Figure)
def detect_figure(obj: Figure):
    return SERIALIZERS["matplotlib"]


try:

    @detect_serializer.register(tq.Circuit)
    @detect_serializer.register(tq.CircuitCollection)
    def detect_trueq(obj: tq.Circuit | tq.CircuitCollection):
        return SERIALIZERS["trueq"]

except NameError:
    ...


def get_serializer(*, key: str | None = None, obj: Any = None) -> Serializer:
    if key is not None:
        normalized = key.lower()
        try:
            return SERIALIZERS[normalized]
        except KeyError:
            ...

    if serializer := detect_serializer(obj):
        return serializer

    registered = ", ".join(f"'{k}'" for k in SERIALIZERS.keys())
    raise KeyError(
        f"'{normalized}' is not a registered serializer. "
        f"Supported values are: {registered}"
    )


register_serializer(DefaultSerializer())
register_serializer(DataFrameSerializer())
register_serializer(ResultSerializer())
register_serializer(MatplotlibSerializer())
register_serializer(TrueQSerializer())

__all__ = [
    "DataFrameSerializer",
    "DefaultSerializer",
    "MatplotlibSerializer",
    "ResultSerializer",
    "Serializer",
    "TrueQSerializer",
    "get_serializer",
    "detect_serializer",
]
