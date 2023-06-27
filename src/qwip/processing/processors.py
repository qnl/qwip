import itertools as it
from collections.abc import Collection
from typing import Generic, TypeVar

import attrs
import numpy as np
import pandas as pd
from attrs import cmp_using, field
from numpy.typing import NDArray
from sklearn.mixture import GaussianMixture
from typing_extensions import Self

from qwip.attrs import _numpy_equals, qdefine
from qwip.processing.data_processor import (
    DATA_PROCESSORS,
    DataProcessor,
    GenericDataProcessor,
    MeasurementResult,
)
from qwip.sequencer import Sequence

M = TypeVar("M", bound=MeasurementResult)


@qdefine
class IQTraceResult(MeasurementResult):
    """Stores raw IQ data vs time step in a multi indexed DataFrame for a single target.

    Time points are assumed to be the same across all elements, readouts, and shots.
    Thus there are N columns corresponding to the field values at the N time steps.
    """

    @classmethod
    def from_numpy(
        cls,
        arr: np.ndarray,
        name: str = "IQTraceResult",
        labels=("element", "readout", "shot"),
        **kwargs,
    ):
        """Creates data frame from trajectory data obtained from QutipBackend.

        Args:
            arr: Complex field amplitudes from a single ADC channel. The default
                indexing is assumed to be (elements, readouts, shots, timepoints), but
                this can be specified by passing in a tuple of labels. The last index
                must always be timepoints.
            name: The result name.
            labels: Index labels for the array axes. These should specify labels for all
                but the last axis.

        Returns:
            An IQ trace result with the data formatted as a multi-indexed dataframe.
        """

        index = pd.MultiIndex.from_tuples(
            it.product(*(range(N) for N in arr.shape[:-1])), names=labels
        )

        data = pd.DataFrame(
            arr.reshape(-1, arr.shape[-1]),
            index=index,
        )

        return cls(name=name, data=data, **kwargs)


@qdefine
class IQResult(MeasurementResult):
    """Demodulated IQ results.

    Attributes:
        data: Multi indexed data frame with row indices (element, readout) and column
        indices (shot) mapping to an I+iQ value.
    """

    @classmethod
    def from_numpy(
        self, arr: np.ndarray, name="IQResult", labels=("element", "readout"), **kwargs
    ) -> Self:
        """Reorders the memory layout of the IQ data for each measurement key.

        Args:
            arr: A numpy array of complex IQ points. The default shape is assumed to be
                (element, readout, shot). The last axis must always be shots.
            name: The result name.
            labels: Index labels for the array axes. These should specify labels for all
                but the last axis.

        Returns:
            An IQResult. The measurement data frame will have index labels as specified,
            and the columns will be the individual shots.
        """

        num_shots = arr.shape[-1]

        index = pd.MultiIndex.from_tuples(
            it.product(*(range(N) for N in arr.shape[:-1])), names=labels
        )
        columns = pd.RangeIndex(num_shots, name="shot")

        df = pd.DataFrame(arr.reshape(-1, num_shots), index=index, columns=columns)

        return IQResult(name=name, data=df)


@DATA_PROCESSORS.register
@qdefine
class HeterodyneDemodulation(DataProcessor):
    """A data processor to demodulate raw IQ traces vs. time.

    Takes in IQTraceResult, a formatted data frame, and demodulates it by
    integrating it with different weighting functions to isolate the information at
    various frequencies specific to readouts.

    Attributes:
        weights: A mapping from readout to the weights used to demodulate the raw
            IQ traces (ex. np.exp(-1j*2pi*freq*ts)).
    """

    weights: dict[str, NDArray[complex]] = field(factory=dict)

    def run(self, result: IQTraceResult, **kwargs) -> list[IQResult]:
        """Demodulates the raw IQ traces at the specified frequencies.

        For now, time points are assumed to be the same across all elements, readouts,
        and shots. Weights later on will be added to account for differences across
        elements and readouts.

        Args:
            result: IQTraceResult.

        Returns:
            A list of IQResult objects containing the processed IQ traces in the `data`
            attribute, where the order matches the weights provided. For the specific
            format of the data frame, go to the IQResult documentation.
        """

        weight_arr = np.stack(list(self.weights.values()))

        integrated = np.dot(weight_arr, result.data.values.T) / weight_arr.shape[-1]

        output = list()
        for i, k in enumerate(self.weights):
            data = pd.DataFrame(integrated[i], index=result.data.index).unstack(-1)

            output.append(IQResult(name=k, data=data))

        return output

    def output_keys(self) -> set[str]:
        """Returns the set of output keys returned by the processor."""
        return set(self.weights)


@DATA_PROCESSORS.register
@qdefine
class IQRotation(DataProcessor):
    """A data processor for rotating IQ data points.

    Attributes:
        angle (float): A phase angle (in radians) to rotate the IQ data by.
    """

    angle: float = 0

    def run(self, result: IQResult, **kwargs) -> IQResult:
        """Rotates the IQ data in the IQ plane by a specified angle.

        Args:
            result: The `IQResult` to rotate.

        Returns:
            The resulting `IQResult` with rotated points.
        """
        angle = kwargs.get("angle", self.angle)

        rotation = np.exp(1j * angle)

        return attrs.evolve(result, data=result.data * rotation)


@qdefine
class ClassifiedResult(MeasurementResult):
    num_states: int = 2
    num_qudits: int = 1


@DATA_PROCESSORS.register
@qdefine
class GMMClassification(DataProcessor):
    """A data processor for classifying IQ data based on a GMM.

    Attributes:
        num_states: The number of qubit states to classify.
        means: The centers of the Gaussian distributions used to model the different
            state distributions. Should have shape `(num_states, 2)`.
        covariances: The covariances of the Gaussian distributiosn used to model the
            different state distributions. We assume spherical Gaussians, so should
            have shape `(num_states,)`.
    """

    num_states: int = 2
    means: np.ndarray = field(eq=cmp_using(_numpy_equals))
    covariances: np.ndarray = field(eq=cmp_using(_numpy_equals))

    @means.default
    def _default_means(self) -> np.ndarray:
        return np.zeros((self.num_states, 2))

    @covariances.default
    def _default_covariances(self) -> np.ndarray:
        return np.ones(self.num_states)

    @staticmethod
    def _get_real_IQ_from_complex(IQ_data: np.ndarray) -> np.ndarray:
        shape = IQ_data.shape

        match IQ_data.dtype:
            case np.complex128:
                cast = np.float64
            case np.complex64:
                cast = np.float32
            case dtype:
                raise TypeError(f"IQ data should be complex type, got {dtype}.")

        return IQ_data.view(cast).reshape(*shape, 2)

    def get_model(self, initialize=True) -> GaussianMixture:
        """Returns an initialized `sklearn.mixture.GaussianMixture` instance.

        This model can be used for classifying IQ points based on the model parameters
        using the `predict` method.

        Args:
            initialize: If `True`, sets the parameters of the `GaussianMixture` object
                to match the means and covariances stored in the data processor.

        Returns:
            An instance of `sklearn.mixture.GaussianMixture`.
        """
        model = GaussianMixture(
            n_components=self.num_states, covariance_type="spherical"
        )

        if initialize:
            model.means_ = self.means
            model.covariances_ = self.covariances
            model.precisions_cholesky_ = 0.01
            model.weights_ = np.ones(self.num_states) / self.num_states

        return model

    def fit(self, result: IQResult, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        """Fits a GMM to the given IQ data.

        Args:
            result: The IQResult whose data is used to fit the GMM.

        Returns:
            A tuple `(means, covariances)` corresponding to the best fit model.
        """
        IQ = GMMClassification._get_real_IQ_from_complex(
            result.data.to_numpy().flatten()
        )

        model = self.get_model()
        model.means_init = self.means
        model.fit(IQ.reshape(-1, 2))

        return model.means_, model.covariances_

    def run(self, result: IQResult, **kwargs) -> ClassifiedResult:
        """Classifies IQData according to the GMM model.

        Args:
            result: The `IQResult` to classify.

        Returns:
            A `ClassifiedResult`. The data will have the same shape as the `IQResult`,
            but with values corresponding to the classified state of the IQ point. Note
            that the values have type `str`.
        """
        model = self.get_model(initialize=True)

        shape = result.shape
        IQ_data = GMMClassification._get_real_IQ_from_complex(
            result.data.to_numpy().flatten()
        )

        classified = model.predict(IQ_data).reshape(shape)
        classified = pd.DataFrame(
            classified, index=result.data.index, columns=result.data.columns, dtype=str
        )

        return ClassifiedResult(
            name=result.name,
            data=classified,
            num_states=self.num_states,
            processors=result.processors,
        )


@DATA_PROCESSORS.register
@qdefine
class ReadoutBitstring(DataProcessor):
    """Concatenates qubit states to determine the joint bitstring per shot.

    Attributes:
        delimiter: The delimiter used to combine measurement keys.
    """

    delimiter: str = ","

    def run(self, result: Collection[ClassifiedResult], **kwargs) -> ClassifiedResult:
        """Concatenates single qudit `ClassifiedResult` into a multi-qudit result.

        Args:
            result: A list of `ClassifiedResult` to concatenate. They must have the same
                shape in order to be combined.

        Returns:
            The multi-qudit `ClassifiedResult`.
        """
        if len(result) == 1:
            return result[0]

        name = self.delimiter.join(m.name for m in result)

        bitstrings = result[0].data.copy()
        for m in result[1:]:
            bitstrings += m.data
        num_states = max(m.num_states for m in result)

        return ClassifiedResult(
            name=name,
            data=bitstrings,
            num_states=num_states,
            num_qudits=len(result),
            processors=tuple(it.chain.from_iterable(m.processors for m in result)),
        )


@qdefine
class HistogramResult(MeasurementResult):
    num_states: int = 2
    num_qudits: int = 1


@DATA_PROCESSORS.register
@qdefine
class ReadoutHistogram(DataProcessor):
    """Bins qudit states/bitstrings to determine counts.

    Attributes:
        fill_missing: Whether or not to fill in missing bitstrings with zero counts.
            This should only be done on small numbers of qudits, since the number of
            possible bitstrings is exponential in the number of qudits.
        sort: Whether or not to sort the bitstring columns. Defaults to `True`.
    """

    fill_missing: bool | None = None
    sort: bool = True

    def run(self, result: ClassifiedResult, **kwargs) -> HistogramResult:
        """Bins qudit states/bitstrings to determine counts.

        Args:
            result: The per-shot bitstrings to bin.

        Returns:
            The resulting histogram data.
        """
        fill_missing = kwargs.get("fill_missing", self.fill_missing)
        if fill_missing is None:
            fill_missing = result.num_qudits <= 1

        # Get information about dataframe shape
        num_ilevels = result.data.index.nlevels
        num_clevels = result.data.columns.nlevels

        index_levels = [i for i in range(num_ilevels)]
        column_levels = [i for i in range(-num_clevels, 0)]

        # Bin by bitstring values
        counts = result.data.stack().groupby(level=index_levels).value_counts()
        counts = counts.unstack(level=column_levels)

        if fill_missing and num_clevels > 1:
            raise ValueError("Fill missing is unsupported for MultiIndex columns")

        elif fill_missing:
            qudit_values = np.arange(result.num_states).astype(str)
            all_bitstrings = [
                "".join(b) for b in it.product(qudit_values, repeat=result.num_qudits)
            ]

            for bitstring in all_bitstrings:
                if bitstring not in counts:
                    counts[bitstring] = 0

        if kwargs.get("sort", self.sort):
            counts.sort_index(axis="columns", inplace=True)

        return HistogramResult(
            name=result.name,
            data=counts.fillna(0),
            num_states=result.num_states,
            num_qudits=result.num_qudits,
            processors=result.processors,
        )


@qdefine
class PopulationResult(MeasurementResult):
    num_states: int = 2
    num_qudits: int = 1
    num_shots: int | None = None


@DATA_PROCESSORS.register
@qdefine
class StatePopulations(DataProcessor):
    """Normalizes bitstring counts to a density."""

    def run(self, result: HistogramResult, **kwargs) -> PopulationResult:
        shots = result.data.sum(axis="columns")

        unique_shots = shots.unique()
        if len(unique_shots) == 1:
            num_shots = unique_shots[0]
        else:
            num_shots = None

        data = result.data.div(shots, axis="index")

        return PopulationResult(
            name=result.name,
            data=data,
            num_states=result.num_states,
            num_qudits=result.num_qudits,
            num_shots=num_shots,
            processors=result.processors,
        )


@DATA_PROCESSORS.register
@qdefine
class Averaged(GenericDataProcessor):
    """A data processor for averaging data along a specified axis.

    Attributes:
        axis (int): The axes along which to average.
    """

    axis: int = 1

    def run(
        self,
        result: M,
        axis: int | None = None,
        level: str | None = None,
        name: str = "averaged",
        **kwargs,
    ) -> M:
        if axis is None:
            axis = self.axis

        if level:
            result.data = result.data.groupby(level, axis=axis).mean()
        else:
            result.data = result.data.mean(axis=axis).to_frame(name=name)

        return result


@DATA_PROCESSORS.register
@qdefine
class Labeled(GenericDataProcessor):
    """A data processor for labeling the results according to the sequence labels."""

    level: str = "element"

    def run(self, result: M, seq: Sequence | None = None, **kwargs) -> M:
        if seq is None:
            return result

        old_idx = result.data.index

        new_idx = pd.DataFrame(
            it.product(
                *(
                    seq.labels.get(n, np.arange(seq.shape[i]))
                    for i, n in enumerate(seq.names)
                )
            ),
            columns=[name or f"{self.level}{i}" for i, name in enumerate(seq.names)],
        ).loc[result.data.index.get_level_values(self.level)]

        for index_level in old_idx.names:
            if index_level != self.level:
                new_idx[index_level] = old_idx.get_level_values(index_level)

        result.data.index = pd.MultiIndex.from_frame(new_idx)

        return result
