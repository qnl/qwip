import itertools as it
from collections.abc import Collection

import attrs
import numpy as np
import pandas as pd
from attrs import field
from sklearn.mixture import GaussianMixture

from qwip.processing.data_processor import (  # register_data_processor
    DATA_PROCESSORS,
    DataProcessor,
    MeasurementResult,
)
from qwip.settings.settings import qdefine


@qdefine
class IQResult(MeasurementResult):
    ...


@DATA_PROCESSORS.register
@qdefine
class FormatLegacyIQ(DataProcessor):
    """A data processor to reformat legacy QTRL IQ data.

    This processor will reorder the axis so that the IQ data for each shot is
    contiguous. The legacy heterodyne array is a 4-D array where the axes correspond
    to `(IQ, shots, elements, readouts)`. This is reformatted to a dataframe where the
    index is `(elements, readouts)` and columns correspond to `shots`, such that the
    data ordering is `(elements, readouts, shots, IQ)`.
    """

    def run(self, meas: np.ndarray, name="IQResult", **kwargs) -> IQResult:
        """Reorders the memory layout of the IQ data for each measurement key.

        Args:
            meas: A `qtrl` measurement dictionary. Each measurement key maps to a 4-d
                numpy array.
            name: The result name.

        Returns:
            An IQResult. The measurement data frame will have index labels
            (element, readout), and the columns will be the individual shots.
        """
        data = np.array(np.transpose(meas, [2, 3, 1, 0]), order="C").view(np.complex128)

        num_se, num_ro, num_shot, _ = data.shape

        index = pd.MultiIndex.from_product(
            [np.arange(num_se), np.arange(num_ro)], names=["element", "readout"]
        )
        columns = pd.RangeIndex(num_shot, name="shot")

        df = pd.DataFrame(data.reshape(-1, num_shot), index=index, columns=columns)

        return IQResult(name=name, data=df)


@DATA_PROCESSORS.register(after=FormatLegacyIQ)
@qdefine
class IQRotation(DataProcessor):
    """A data processor for rotating IQ data points.

    Attributes:
        angle (float): A phase angle (in radians) to rotate the IQ data by.
    """

    angle: float = 0

    def run(self, meas: IQResult, **kwargs) -> IQResult:
        angle = kwargs.get("angle", self.angle)

        rotation = np.exp(1j * angle)

        return attrs.evolve(meas, data=meas.data * rotation)


# @DATA_PROCESSORS.register
# @qdefine
# class AverageIQ(DataProcessor):
#     """A data processor for averaging IQ data points.

#     Attributes:
#         axis (int): The axes along which to average.
#     """
#     axis: int = -1

#     def run(self, meas: IQResult, **kwargs) -> IQResult:
#         raise NotImplementedError()


@qdefine
class ClassifiedResult(MeasurementResult):
    num_states: int = 2
    num_qudits: int = 1


@DATA_PROCESSORS.register
@qdefine
class GMMClassification(DataProcessor):
    num_states: int = 2
    means: np.ndarray = field()
    covariances: np.ndarray = field()

    @means.default
    def _default_means(self) -> np.ndarray:
        return np.zeros((self.num_states, 2))

    @covariances.default
    def _default_covariances(self) -> np.ndarray:
        return np.ones(self.num_states)

    def get_model(self, initialize=True) -> GaussianMixture:
        model = GaussianMixture(
            n_components=self.num_states, covariance_type="spherical"
        )

        if initialize:
            model.means_ = self.means
            model.covariances_ = self.covariances
            model.precisions_cholesky_ = 0.01
            model.weights_ = np.ones(self.num_states) / self.num_states

        return model

    def run(self, meas: IQResult, **kwargs) -> ClassifiedResult:
        model = self.get_model(initialize=True)

        shape = meas.data.shape
        IQ_data = meas.data.to_numpy().view(np.float64).reshape(-1, 2)

        classified = model.predict(IQ_data).reshape(shape)
        classified = pd.DataFrame(
            classified, index=meas.data.index, columns=meas.data.columns, dtype=str
        )

        return ClassifiedResult(
            name=meas.name,
            data=classified,
            num_states=self.num_states,
            processors=meas.processors,
        )


@DATA_PROCESSORS.register
@qdefine
class ReadoutBitstring(DataProcessor):
    delimiter: str = ","

    def run(self, meas: Collection[ClassifiedResult], **kwargs) -> ClassifiedResult:
        if len(meas) == 1:
            return meas[0]

        name = self.delimiter.join(m.name for m in meas)

        bitstrings = meas[0].data.copy()
        for m in meas[1:]:
            bitstrings += m.data
        num_states = max(m.num_states for m in meas)

        return ClassifiedResult(
            name=name,
            data=bitstrings,
            num_states=num_states,
            num_qudits=len(meas),
            processors=tuple(it.chain.from_iterable(m.processors for m in meas)),
        )


@qdefine
class HistogramResult(MeasurementResult):
    num_states: int = 2
    num_qudits: int = 1


@DATA_PROCESSORS.register
@qdefine
class ReadoutHistogram(DataProcessor):
    fill_missing: bool | None = None
    sort: bool = True

    def run(self, meas: ClassifiedResult, **kwargs) -> HistogramResult:
        fill_missing = kwargs.get("fill_missing", self.fill_missing)
        if fill_missing is None:
            fill_missing = meas.num_qudits <= 1

        # Get information about dataframe shape
        num_ilevels = meas.data.index.nlevels
        num_clevels = meas.data.columns.nlevels

        index_levels = [i for i in range(num_ilevels)]
        column_levels = [i for i in range(-num_clevels, 0)]

        # Bin by bitstring values
        counts = meas.data.stack().groupby(level=index_levels).value_counts()
        counts = counts.unstack(level=column_levels)

        if fill_missing and num_clevels > 1:
            raise ValueError("Fill missing is unsupported for MultiIndex columns")

        elif fill_missing:
            qudit_values = np.arange(meas.num_states).astype(str)
            all_bitstrings = [
                "".join(b) for b in it.product(qudit_values, repeat=meas.num_qudits)
            ]

            for bitstring in all_bitstrings:
                if bitstring not in counts:
                    counts[bitstring] = 0

        if kwargs.get("sort", self.sort):
            counts.sort_index(axis="columns", inplace=True)

        return HistogramResult(
            name=meas.name,
            data=counts.fillna(0),
            num_states=meas.num_states,
            num_qudits=meas.num_qudits,
            processors=meas.processors,
        )


@qdefine
class PopulationResult(MeasurementResult):
    num_states: int = 2
    num_qudits: int = 1
    num_shots: int | None = None


@DATA_PROCESSORS.register
@qdefine
class StatePopulations(DataProcessor):
    fill_missing: bool | None = None
    sort: bool = True

    def run(self, meas: HistogramResult, **kwargs) -> PopulationResult:
        shots = meas.data.sum(axis="columns")

        unique_shots = shots.unique()
        if len(unique_shots) == 1:
            num_shots = unique_shots[0]
        else:
            num_shots = None

        data = meas.data.div(shots, axis="index")

        return PopulationResult(
            name=meas.name,
            data=data,
            num_states=meas.num_states,
            num_qudits=meas.num_qudits,
            num_shots=num_shots,
            processors=meas.processors,
        )
