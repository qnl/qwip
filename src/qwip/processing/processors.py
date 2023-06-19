import itertools as it
from collections.abc import Collection

import attrs
import numpy as np
import pandas as pd
from attrs import cmp_using, field
from numpy.typing import NDArray
from sklearn.mixture import GaussianMixture

from qwip.attrs import qdefine
from qwip.processing.data_processor import (  # register_data_processor
    DATA_PROCESSORS,
    DataProcessor,
    MeasurementResult,
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
        name: str,
        IQ_raw: np.ndarray,
        labels=("element", "readout", "shot"),
        **kwargs,
    ):
        """Creates data frame from trajectory data obtained from QutipBackend.

        Args:
            name: The name of the `IQTraceResult`.
            IQ_raw: Complex field amplitudes from a single ADC channel. The default
                indexing is assumed to be (elements, readouts, shots, timepoints), but
                this can be specified by passing in a tuple of labels. The last index
                must always be timepoints.
            labels: Index labels for the array axes. These should specify labels for all
                but the last axis.


        Returns:
            An IQ trace result with the data formatted as a multi-indexed dataframe.
        """

        index = pd.MultiIndex.from_tuples(
            it.product(*(range(N) for N in IQ_raw.shape[:-1])), names=labels
        )

        data = pd.DataFrame(
            IQ_raw.reshape(-1, IQ_raw.shape[-1]),
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

    data: pd.DataFrame


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

    def run(self, meas: IQTraceResult, **kwargs) -> dict[str, IQResult]:
        """Demodulates the raw IQ traces at the specified frequencies.

        For now, time points are assumed to be the same across all elements, readouts,
        and shots. Weights later on will be added to account for differences across
        elements and readouts.

        Args:
            meas: IQTraceResult.

        Returns:
            A dictionary mapping readout to its IQResult object containing the processed
            IQ traces in the `data` attribute. For the specific format of the data frame,
            go to the IQResult documentation.
        """

        weight_arr = np.stack(self.weights.values())

        integrated = np.dot(weight_arr, meas.data.values.T) / weight_arr.shape[-1]

        result = dict()
        for i, k in enumerate(self.weights):
            data_k = pd.DataFrame(integrated[i], index=meas.data.index).unstack(-1)

            result[k] = IQResult(name=f"{meas.name}_{k}", data=data_k)

        return result


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
        match meas.dtype:
            case np.float32:
                cast = np.complex64
            case np.float64:
                cast = np.complex128
            case _:
                meas = meas.astype(float)
                cast = np.complex128

        data = np.array(np.transpose(meas, [2, 3, 1, 0]), order="C").view(cast)

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
        """Rotates the IQ data in the IQ plane by a specified angle.

        Args:
            meas: The `IQResult` to rotate.

        Returns:
            The resulting `IQResult` with rotated points.
        """
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
    means: np.ndarray = field()
    covariances: np.ndarray = field()

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
                TypeError(f"IQ data should be complex type, got {dtype}.")

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

    def fit(self, meas: IQResult, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        """Fits a GMM to the given IQ data.

        Args:
            meas: The IQResult whose data is used to fit the GMM.

        Returns:
            A tuple `(means, covariances)` corresponding to the best fit model.
        """
        IQ = GMMClassification._get_real_IQ_from_complex(meas.data.to_numpy().flatten())

        model = self.get_model()
        model.means_init = self.means
        model.fit(IQ.reshape(-1, 2))

        return model.means_, model.covariances_

    def run(self, meas: IQResult, **kwargs) -> ClassifiedResult:
        """Classifies IQData according to the GMM model.

        Args:
            meas: The `IQResult` to classify.

        Returns:
            A `ClassifiedResult`. The data will have the same shape as the `IQResult`,
            but with values corresponding to the classified state of the IQ point. Note
            that the values have type `str`.
        """
        model = self.get_model(initialize=True)

        shape = meas.shape
        IQ_data = GMMClassification._get_real_IQ_from_complex(
            meas.data.to_numpy().flatten()
        )

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
    """Concatenates qubit states to determine the joint bitstring per shot.

    Attributes:
        delimiter: The delimiter used to combine measurement keys.
    """

    delimiter: str = ","

    def run(self, meas: Collection[ClassifiedResult], **kwargs) -> ClassifiedResult:
        """Concatenates single qudit `ClassifiedResult` into a multi-qudit result.

        Args:
            meas: A list of `ClassifiedResult` to concatenate. They must have the same
                shape in order to be combined.

        Returns:
            The multi-qudit `ClassifiedResult`.
        """
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
    """Bins qudit states/bitstrings to determine counts.

    Attributes:
        fill_missing: Whether or not to fill in missing bitstrings with zero counts.
            This should only be done on small numbers of qudits, since the number of
            possible bitstrings is exponential in the number of qudits.
        sort: Whether or not to sort the bitstring columns. Defaults to `True`.
    """

    fill_missing: bool | None = None
    sort: bool = True

    def run(self, meas: ClassifiedResult, **kwargs) -> HistogramResult:
        """Bins qudit states/bitstrings to determine counts.

        Args:
            meas: The per-shot bitstrings to bin.

        Returns:
            The resulting histogram data.
        """
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
    """Normalizes bitstring counts to a density."""

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
