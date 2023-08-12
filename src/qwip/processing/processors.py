import itertools as it
from collections.abc import Collection
from typing import Generic, TypeVar

import attrs
import numpy as np
import pandas as pd
from attrs import cmp_using, field
from numpy.random import Generator, default_rng
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

RESULT_RNG = default_rng()


def array_complex_to_real(arr: np.ndarray) -> np.ndarray:
    """Converts a complex-valued array to a real-valued array.

    The numpy array is view casted to a real array where the last axis contains the
    real and imaginary components. If the original array is C-contiguous in memory, no
    copy is performed.

    Args:
        arr: The array to convert.

    Returns:
        A real-valued array with the real and imaginary components.
    """
    shape = arr.shape

    match arr.dtype:
        case np.complex128:
            cast = np.float64
        case np.complex64:
            cast = np.float32
        case dtype:
            raise TypeError(f"Array should be complex type, got {dtype}.")

    return np.ascontiguousarray(arr).view(cast).reshape(*shape, 2)


def array_real_to_complex(arr: np.ndarray) -> np.ndarray:
    """Converts a real-valued array to a complex-valued array.

    The numpy array is view casted to a complex-valued array. If the original array is
    C-contiguous in memory, no copy is performed.

    Args:
        arr: The array to convert.

    Returns:
        A complex-valued array.
    """
    match arr.dtype:
        case np.float64:
            cast = np.complex128
        case np.float32:
            cast = np.complex64
        case dtype:
            raise TypeError(f"Array should be float type, got {dtype}.")

    return np.ascontiguousarray(arr).view(cast)


def dataframe_complex_to_real(
    data: pd.DataFrame, names: tuple[str, str] = ("real", "imag")
) -> pd.DataFrame:
    """Splits complex columns into real and imaginary columns.

    The dataframe must consist of only complex columns.

    Args:
        data: The dataframe to convert.
        names: The column names for the real and complex components.

    Returns:
        A new dataframe with complex columns replaced by a real and imaginary column.

    Raises:
        ValueError: If the dataframe columns are not compatible.
    """

    dtypes = data.dtypes.unique()
    if len(dtypes) > 1 or not issubclass(dtypes[0].type, (complex, np.complexfloating)):
        raise ValueError(
            f"Cannot convert dataframe with non-complex columns. Got {data.dtypes}"
        )

    columns = []
    for levels in data.columns:
        if data.columns.nlevels == 1:
            levels = (levels,)
        columns.append((*levels, names[0]))
        columns.append((*levels, names[1]))

    columns = pd.MultiIndex.from_tuples(columns, name=[*data.columns.names, "complex"])
    return pd.DataFrame(
        array_complex_to_real(data.values).reshape(data.values.shape[0], -1),
        index=data.index,
        columns=columns,
    )


def dataframe_real_to_complex(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Combines a dataframe with real and imaginary columns into complex columns.

    The dataframe must consist of only float columns.

    Args:
        data: The dataframe to convert.

    Returns:
        A new dataframe with real and imaginary columns replaced by a single complex
        column.

    Raises:
        ValueError: If the dataframe columns are not compatible.
    """

    dtypes = data.dtypes.unique()
    if len(dtypes) > 1 or not issubclass(dtypes[0].type, (float, np.floating)):
        raise ValueError(
            f"Cannot convert dataframe with non-float columns. Got {data.dtypes}"
        )

    columns = data.columns.droplevel(-1).unique()
    return pd.DataFrame(
        array_real_to_complex(data.values),
        index=data.index,
        columns=columns,
    )


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
    ) -> Self:
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
        data: Multi indexed dataframe indexed by (element, shot, readout).
    """

    @classmethod
    def from_numpy(
        cls,
        arr: np.ndarray,
        name="IQResult",
        labels=("element", "shot", "readout"),
        **kwargs,
    ) -> Self:
        """Reorders the memory layout of the IQ data for each measurement key.

        Args:
            arr: A numpy array of complex IQ points. The default shape is assumed to be
                (element, shot, readout).
            name: The result name.
            labels: Index labels for the array axes. These should specify labels for all
                but the last axis.

        Returns:
            An IQResult. The measurement data frame will have index labels as specified,
            and the columns will be the individual shots.
        """
        index = pd.MultiIndex.from_tuples(
            it.product(*(range(N) for N in arr.shape)), names=labels
        )

        df = pd.DataFrame(arr.flatten(), index=index, columns=["IQ"])

        return cls(name=name, data=df)

    @classmethod
    def random(
        cls,
        shape: tuple[int, ...],
        num_states: int = 2,
        rng: Generator = RESULT_RNG,
        **kwargs,
    ) -> Self:
        """Creates a random dataframe with the given shape.

        Args:
            shape: The shape of the resulting array. The default axis are
                `(element, shot, readout)`. If a different number of axes are passed, a
                set of labels should also be specified.
            num_states: The number of "blobs" to generate. The means and standard
                deviations of the Gaussian "blobs" are chosen randomly.
            rng: The random generator to use.

        Returns:
            An IQResult populated with random data.
        """
        states = rng.choice(num_states, size=np.prod(shape)).reshape(shape)
        means = rng.normal(scale=10, size=2 * num_states).reshape(num_states, 2)
        iqdata = np.zeros_like(states, dtype=np.complex64)

        for s in range(num_states):
            iq = (
                rng.multivariate_normal(
                    mean=means[s],
                    cov=np.identity(2),
                    size=np.count_nonzero(states == s),
                )
                .astype(np.float32)
                .reshape(-1, 2)
            )
            iqdata[states == s] = array_real_to_complex(iq).flatten()

        return cls.from_numpy(iqdata)


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
            A dictionary mapping readout to its IQResult object containing the processed
            IQ traces in the `data` attribute. For the specific format of the data frame,
            go to the IQResult documentation.
        """

        weight_arr = np.stack(list(self.weights.values()))

        integrated = np.dot(weight_arr, result.data.values.T) / weight_arr.shape[-1]

        output = list()
        for i, k in enumerate(self.weights):
            data = pd.Series(integrated[i], index=result.data.index).unstack(-1)

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

    @classmethod
    def from_numpy(
        cls,
        arr: np.ndarray,
        name="ClassifiedResult",
        labels=("element", "shot", "readout"),
        **kwargs,
    ) -> Self:
        """Creates a `ClassifiedResult` instance from a numpy array of states.

        Args:
            arr: A numpy array of complex IQ points. The default shape is assumed to be
                (element, shot, readout).
            name: The result name.
            labels: Index labels for the array axes. These should specify labels for all
                but the last axis.
            **kwargs: Remaining keyword arguments are passed to the init method.

        Returns:
            An ClassifiedResult. The measurement dataframe will have index labels as
            specified, with a single column with all the states.
        """
        index = pd.MultiIndex.from_tuples(
            it.product(*(range(N) for N in arr.shape)), names=labels
        )

        arr = np.asarray(arr, dtype=int).flatten()
        df = pd.DataFrame(arr, index=index, columns=["state"], dtype=str)

        return cls(name=name, data=df, **kwargs)


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
        IQ = array_complex_to_real(result.data.to_numpy().flatten())

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
        IQ_data = array_complex_to_real(result.data.to_numpy().flatten())

        classified = model.predict(IQ_data).reshape(shape)
        classified = pd.DataFrame(
            classified, index=result.data.index, columns=["state"], dtype=str
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
            return attrs.evolve(result[0])

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
        index_levels = result.data.index.names
        num_ilevels = result.data.index.nlevels
        num_clevels = result.data.columns.nlevels

        stack_levels = list(range(num_clevels - 1))
        group_levels = [i for i, n in enumerate(index_levels) if n != "shot"] + [
            num_ilevels + i for i in stack_levels
        ]
        unstack_levels = list(range(-num_clevels, 0))

        # Bin by bitstring values
        counts = result.data.stack(stack_levels)
        counts = counts.groupby(level=group_levels).value_counts()
        counts = counts.unstack(unstack_levels)

        if fill_missing:
            if num_clevels > 1:
                raise ValueError("Fill missing is unsupported for MultiIndex columns")

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
        if result.data.columns.nlevels > 1:
            column_levels = result.data.columns.names
            groupby_levels = [i for i, n in enumerate(column_levels) if n != "state"]
            shots = result.data.groupby(level=groupby_levels, axis="columns").sum()
        else:
            shots = result.data.sum(axis="columns")

        unique_shots = np.unique(shots)
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

    axis: int = 0
    level: str = "shot"

    def run(
        self,
        result: M,
        axis: int | None = None,
        level: str | None | type(...) = None,
        name: str = "averaged",
        **kwargs,
    ) -> M:
        if axis is None:
            axis = self.axis

        if level is None:
            level = self.level

        result = attrs.evolve(result)

        all_levels = getattr(result.data, "columns" if axis else "index").names
        if level is ... or level not in all_levels:
            result.data = result.data.mean(axis=axis).to_frame(name=name)
        else:
            result.data = result.data.groupby(
                [n for n in all_levels if n != level], axis=axis
            ).mean()

        return result


@DATA_PROCESSORS.register
@qdefine
class Labeled(GenericDataProcessor):
    """A data processor for labeling the results according to the sequence labels."""

    level: str = "element"

    def run(self, result: M, seq: Sequence | None = None, **kwargs) -> M:
        if seq is None:
            return result

        result = attrs.evolve(result, data=result.data.copy())
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
