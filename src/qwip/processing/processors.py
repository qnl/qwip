import itertools as it
from collections.abc import Collection
from typing import Any, TypeVar

import attrs
import numpy as np
import pandas as pd
from attrs import cmp_using, field
from loguru import logger
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
from qwip.sequencer.compilation import QuantumExecutable

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

    Time points are assumed to be the same across all timelines, readouts, and shots.
    Thus there are N columns corresponding to the field values at the N time steps.
    """

    @classmethod
    def from_numpy(
        cls,
        arr: np.ndarray,
        name: str = "IQTraceResult",
        labels: tuple[str, ...] = ("shot", "timeline", "readout", "time"),
        **kwargs: Any,
    ) -> Self:
        """Creates data frame from trajectory data obtained from QutipBackend.

        Args:
            arr: Complex field amplitudes from a single ADC channel. The default
                indexing is assumed to be (shots, timelines, readouts, timepoints), but
                this can be specified by passing in a tuple of labels. The last index
                must always be timepoints.
            name: The result name.
            labels: Index labels for the array axes. These should specify labels for all
                but the last axis.

        Returns:
            An IQ trace result with the data formatted as a multi-indexed dataframe.
        """

        index = pd.MultiIndex.from_arrays(
            np.indices(arr.shape).reshape(len(arr.shape), -1), names=labels
        )

        data = pd.DataFrame(
            arr.flatten(),
            index=index,
        )

        return cls(name=name, data=data, **kwargs)


@qdefine
class IQResult(MeasurementResult):
    """Demodulated IQ results.

    Attributes:
        data: Multi indexed dataframe indexed by (timeline, shot, readout).
    """

    @classmethod
    def from_numpy(
        cls,
        arr: np.ndarray,
        name: str = "IQResult",
        labels: tuple[str, ...] = ("shot", "timeline", "readout"),
        **kwargs: Any,
    ) -> Self:
        """Reorders the memory layout of the IQ data for each measurement key.

        Args:
            arr: A numpy array of complex IQ points. The default shape is assumed to be
                (shot, timeline, readout).
            name: The result name.
            labels: Index labels for the array axes. These should specify labels for all
                but the last axis.

        Returns:
            An IQResult. The measurement data frame will have index labels as specified,
            and the columns will be the individual shots.
        """
        index = pd.MultiIndex.from_arrays(
            np.indices(arr.shape).reshape(len(arr.shape), -1), names=labels
        )

        df = pd.DataFrame(arr.flatten(), index=index, columns=["IQ"])

        return cls(name=name, data=df, **kwargs)

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
                `(shot, timeline, readout)`. If a different number of axes are passed, a
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

        return cls.from_numpy(iqdata, **kwargs)

    def amplitude(self, log: bool = False) -> pd.DataFrame:
        """Computes the amplitude of the IQ data.

        Args:
            log: If true, computes the amplitude in units of dB (Power)

        Returns:
            A dataframe with the amplitude of the IQ data.
        """
        amp = self.data.abs()

        if len(amp.columns) == 1:
            amp.columns = ["amplitude"]

        return 20 * np.log10(amp) if log else amp

    def phase(
        self,
        *,
        unwrap: bool = False,
        electrical_delay: float = 0,
        frequency: str | int = "frequency",
    ) -> pd.DataFrame:
        """Computes the phase of the IQ data.

        Args:
            unwrap: Whether or not to unwrap the phase.
            electrical_delay: The electrical delay, in seconds, that should be removed
                from the phase. If electrical delay is nonzero, the index level for
                frequency must be supplied.
            frequency: The name of the index level corresponding to a frequency. This is
                necessary only if a non-zero electrical delay is being applied. An `int`
                specifying the index level can also be given.

        Returns:
            A dataframe with the phase of the IQ data.
        """
        if electrical_delay:
            data = self.electrical_delay(electrical_delay, frequency=frequency).data
        else:
            data = self.data

        phase = np.angle(data)
        if unwrap:
            phase = np.unwrap(phase, axis=0)

        cols = ["phase"] if len(self.data.columns) == 1 else self.data.columns
        return pd.DataFrame(phase, columns=cols, index=self.data.index)

    def electrical_delay(
        self,
        delay: float = 0,
        frequency: str | int = "frequency",
        inplace: bool = False,
    ) -> Self:
        """Applies an electrical delay to the IQ data.

        Args:
            delay: The electrical delay, in seconds to apply.
            frequency: The name of the index level corresponding to a frequency. This is
                necessary only if a non-zero electrical delay is being applied. An `int`
                specifying the index level can also be given.
            inplace: If `True`, the result data is modified inplace. Otherwise, a new
                result is created and returned.

        Returns:
            A new result with delay applied to the result data, or the modified result
            if `inplace=True`.
        """
        fs = self.data.index.get_level_values(frequency).to_numpy()
        delay = np.exp(1j * 2 * np.pi * fs * delay).reshape(-1, 1)

        if inplace:
            self.data *= delay
            return self
        else:
            data = self.data * delay
            return attrs.evolve(self, data=data)


@DATA_PROCESSORS.register
@qdefine
class HeterodyneDemodulation(DataProcessor):
    """A data processor to demodulate raw IQ traces vs. time.

    Takes in IQTraceResult, a formatted data frame, and demodulates it by
    integrating it with different weighting functions to isolate the information at
    various frequencies specific to readouts.

    Attributes:
        device: The name of the device/program that stores the weights.
    """

    device: str = "demod"

    def run(
        self, result: IQTraceResult, exe: QuantumExecutable, **kwargs
    ) -> list[IQResult]:
        """Demodulates the raw IQ traces at the specified frequencies.

        The demodulated channels may not be the same across all timelines/readouts, in
        which case the resulting IQ values at those index locations will be nan. The
        demodulation weights can also vary for each readout. See `HeterodyneProgram` for
        more details.

        Args:
            result: An IQTraceResult. The index should have the same number timesteps
                for every trace. It is assumed that the index levels are of the form
                `(shot, timeline, readout, time)`.

        Returns:
            A list of IQResult, one for each demodulation channel present.
        """

        in_shape = result.data.index.levshape
        n_shots = in_shape[0]
        n_times = in_shape[-1]
        arr = result.data.values.reshape(n_shots, -1, n_times)
        arr = arr - np.mean(arr)

        if arr.dtype != np.complex64:
            logger.warning(f"IQ trace data has dtype {arr.dtype}!")

        program = exe.programs[self.device]
        outputs = {
            key: np.zeros(np.prod(in_shape[:-1]), dtype=arr.dtype)
            for key in program.keys
        }

        for w_idx, weights in enumerate(program.weights):
            (dslice,) = np.where(program.demods == w_idx)

            t_cutoff = min(n_times, weights.shape[-1])
            integrated = (
                np.dot(
                    weights[:, :t_cutoff].conj(),
                    arr[:, dslice, :t_cutoff].reshape(-1, t_cutoff).T,
                )
                / t_cutoff
            )

            for col, key in enumerate(program.keys):
                outputs[key].reshape(n_shots, -1)[:, dslice] = integrated[col].reshape(
                    n_shots, -1
                )

        out_index = result.data.index[::n_times].droplevel(-1)
        return [
            IQResult(name=key, data=pd.DataFrame(data, index=out_index, columns=["IQ"]))
            for key, data in outputs.items()
        ]

    def output_keys(self) -> set[str]:
        """Returns the set of output keys returned by the processor."""
        return {...}


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
        name: str = "ClassifiedResult",
        labels: tuple[str, ...] = ("shot", "timeline", "readout"),
        **kwargs: int,
    ) -> Self:
        """Creates a `ClassifiedResult` instance from a numpy array of states.

        Args:
            arr: A numpy array of complex IQ points. The default shape is assumed to be
                (timeline, shot, readout).
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

    def get_model(self, initialize: bool = True) -> GaussianMixture:
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
        model.means_init = kwargs.get("init_means", self.means)
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

    level: str = "timeline"

    def run(self, result: M, exe: QuantumExecutable | None = None, **kwargs) -> M:
        if exe is None or exe.seq is None:
            return result

        seq = exe.sequence
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
        )
        broadcasted = new_idx.loc[result.data.index.get_level_values(self.level)]

        idx_vals = []
        idx_names = []
        for name in old_idx.names:
            if name == self.level:
                for c in new_idx.columns:
                    idx_vals.append(broadcasted[c].values)
                    idx_names.append(c)
            else:
                level = old_idx.get_level_values(name)
                idx_vals.append(level.values)
                idx_names.append(level.name)

        result.data.index = pd.MultiIndex.from_arrays(idx_vals, names=idx_names)

        return result


__all__ = [
    "array_complex_to_real",
    "array_real_to_complex",
    "dataframe_complex_to_real",
    "dataframe_real_to_complex",
    "Averaged",
    "ClassifiedResult",
    "GMMClassification",
    "HeterodyneDemodulation",
    "IQResult",
    "IQRotation",
    "IQTraceResult",
    "Labeled",
    "PopulationResult",
    "ReadoutBitstring",
    "ReadoutHistogram",
    "StatePopulations",
]
