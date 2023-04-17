from typing import Mapping, Optional

import numpy as np
import pandas as pd
import pendulum
from attr import field
from loguru import logger
from numpy.typing import NDArray
from qtrl.am_seq_utils.am_sequences import add_basic_readout
from qtrl.amiable_sequencer import AM_Sequence
from sklearn.mixture import GaussianMixture

# from qwip.processing.classification import GMMData
from qwip.attrs import qdefine
from qwip.calibration.calibration import Calibration, CalibrationLogger, Parameter
from qwip.flatdict import FlatDict

# from qwip.visualization.readout import plot_decision_boundary, plot_readout_histogram


@qdefine
class ReadoutCalibration(Calibration):
    @qdefine
    class Result(Calibration.Result):
        means: NDArray[np.float_]
        covariances: NDArray[np.float_]
        separations: FlatDict[str, float] = field(factory=FlatDict)

        def gmm_model(self):
            n_states = self.means.shape[0]

            mix = GaussianMixture(n_components=n_states, covariance_type="spherical")
            mix.means_ = self.means
            mix.covariances_ = self.covariances
            mix.precisions_cholesky_ = 0.01
            mix.weights_ = np.ones(n_states) / n_states

            return mix

    parameter: str = "gmm"

    def get_sequence(self, config, qubits):
        seq = AM_Sequence(shape=2, sequence_source=config.am_pulses)
        prev_end = "start"
        for q in qubits:
            seq[1].append(f"Q{q}/X180", prev_end, seq_name=f"Q{q}_pi")
            prev_end = f"Q{q}_pi/end"
        seq[1].set_location("end", prev_end)
        seq[0].set_location("end", "start")

        add_basic_readout(seq, config, ro_qubits=qubits)

        return seq

    def analysis(self, data, num_states=2):
        output = FlatDict()

        ndict = dict(default=2)

        if isinstance(num_states, Mapping):
            ndict.update(num_states)
        else:
            ndict["default"] = num_states

        for key, IQ in data.items():
            N = ndict.get(key, ndict["default"])
            gmm = GaussianMixture(n_components=N, covariance_type="spherical").fit(
                IQ.reshape(-1, 2)
            )

            means = gmm.means_
            covs = gmm.covariances_
            seps = self.get_separation(means, covs, N=N)
            output[key] = ReadoutCalibration.Result(
                means=means, covariances=covs, separations=seps
            )

        self.results.update(output)

        return output

    def plot(self, data, results):
        fig = plot_readout_histogram(data)
        fig = plot_decision_boundary(fig, results)
        return fig

    def get_separation(self, means, covariances, N=2):
        separations = {}
        for pair in np.array(np.triu_indices(N, 1)).T:
            separation = np.sqrt(np.sum(np.diff(means[pair], axis=1) ** 2))
            std = np.product(covariances[pair]) ** 0.25
            separations["{0}-{1}".format(*pair)] = separation / std

        return separations

    def to_dataframe(self, results):
        rows = []
        index = []
        for key in results.flatkeys():
            N = len(results[key].means)
            rows.append(
                np.hstack([results[key].means, results[key].covariances[:, np.newaxis]])
            )
            index += [(key, state) for state in range(N)]

        index = pd.MultiIndex.from_tuples(index)
        index.names = ["key", "state"]
        df = pd.DataFrame(
            np.vstack(rows), index=index, columns=["xmean", "ymean", "variance"]
        )

        timestamps = pd.DataFrame.from_dict(
            {
                k: r.timestamp.isoformat(timespec="seconds")
                for k, r in results.flatitems()
            },
            orient="index",
            columns=["timestamp"],
        )
        timestamps.index.name = "key"

        return df.join(timestamps)
