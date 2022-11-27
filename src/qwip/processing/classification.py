"""Processing blocks related to classification"""

from typing import Mapping, Optional, Set, Union, Annotated

import numpy as np

from loguru import logger
from attr import field
from sklearn.mixture import GaussianMixture

from qwip.typing import NDArray
from qwip.defaults import dynamic_default
from qwip.flatdict import FlatDict
from qwip.settings.settings import qdefine, Settings
from qwip.processing.process import Process

@qdefine
class IQRotation(Process):
    """A processing block for rotating heterodyne data in the IQ plane.
    
    This process expects a mapping (dict) of strings to ndarays as its input. The
    keys are typically of the form R(\\d+), but can be anything. Each ndarray is
    expected to have the shape `(..., IQ)`.

    Attributes:
        angle (dict): A dictionary of angles for each data key. The unit defaults
            to `qsettings['units/angle']` but can be set as a keword argument
            to `run()`.

    """
    angles: dict[str, float] = field(factory=dict)

    @dynamic_default(unit='units/angle')
    def run(self, data, /, unit: str = None):
        """Rotates the data by the specified angle.
        
        Args:
            data (dict): a mapping of names to numpy arrays containing the
                heterodyne data.
        """

        output = {}

        for key, IQ in data.items():
            if key not in self.angles:
                logger.warning(f"No rotation angle for '{key}'.")

            angle = self.angles.get(key, 0)

            if unit == 'degrees':
                angle *= np.pi / 180
        
            rotation = np.exp(1j*angle)

            output[key] = (rotation * IQ.view(complex)).view(float)

        return output

@qdefine
class GMMData(Settings):
    means: NDArray[np.float64]
    covariances: NDArray[np.float64]
    separations: FlatDict[str, float] = field(factory=FlatDict)

    def gmm_model(self):
        n_states = self.means.shape[0]

        mix = GaussianMixture(n_components=n_states, covariance_type='spherical') 
        mix.means_ = self.means
        mix.covariances_ = self.covariances
        mix.precisions_cholesky_ = 0.01
        mix.weights_ = np.ones(n_states) / n_states

        return mix

@qdefine
class GMM(Process):
    """A processing block for GMM classficiation of heterodyne data.
    
    This process expects a mapping (dict) of strings to ndarrays as its input. The
    keys are typically of the form R(\\d+), but can be anything. Each ndarray is
    expected to have the shape `(..., IQ)`.
    """

    gmms: FlatDict[str, GMMData]
    mixes: FlatDict[str, GaussianMixture] = field(init=False, metadata=dict(serialize=False))

    def run(self, data, /):
        """Classifies the data using a Gaussian Mixture model.

        This processing block uses `sklearn.mixture.GaussianMixture` to implement
        the predictions.
        """
        output = {}

        for key, IQ in data.items():
            if key not in self.gmms:
                logger.error(f"No '{key}'.")
                raise KeyError(
                    f"Cannot classify IQ data for key '{key}' with no specified"
                    f" GMMs."
                )

            elif key not in self.mixes:
                self.mixes = self.generate_mixes()

            mix = self.mixes[key]
            

            # These reshapes only create views of the data where memory is
            # still contiguous
            classified_shape = IQ.shape[:-1]
            classified = mix.predict(IQ.reshape((-1, 2)))
            output[key] = classified.reshape(classified_shape)

        return output

    @mixes.default
    def generate_mixes(self):
        mixes = FlatDict()

        for k, result in self.gmms.items():
            mixes[k] = result.gmm_model()

        return mixes

    # def _validate_means_covariances(self):
    #     ms = set(self.means.flatkeys())
    #     cs = set(self.covariances.flatkeys())

    #     if no_covs := ms - cs:
    #         key_str = ', '.join(no_covs)
    #         raise ValueError(
    #             f'Mismatch between means and covariances! The following keys '
    #             f'have no covariances: {key_str}')

    #     if no_means := cs - ms:
    #         key_str = ', '.join(no_means)
    #         raise ValueError(
    #             f'Mismatch between means and covariances! The following keys '
    #             f'have no means: {key_str}')

    #     for key in ms:
    #         if self.means[key].shape[0] != self.covariances[key].shape[0]:
    #             raise ValueError(
    #                 f"Mismatch between means and covariances for '{key}'. Means"
    #                 f" have shape {self.means[key].shape} but covariances have "
    #                 f"shape {self.covariances[key].shape}"
    #             )

@qdefine
class StatePopulations(Process):
    """A processing block for getting state populations from classified data.

    This process expects a mapping (dict) of strings to ndarrays as its input. The
    keys are typically of the form R(\\d+), but can be anything. Each ndarray is
    expected to have the shape `(n_shots, ...)`.
    """

    states: Union[int, dict[str, int]] = field(default=2, metadata=dict(auto_convert=False))
    axis: int = 0

    def run(self, data, /):
        """Gets averaged state populations from classified data.

        This processing block uses `sklearn.mixture.GaussianMixture` to implement
        the predictions.
        """

        outputs = {}
        
        for key, classified in data.items():
            default_n = max(2, np.max(classified) + 1)

            if isinstance(self.states, int):
                n = self.states
            else:
                n = self.states.get(key, default=self.states.get('default'))
                if n is None:
                    n = default_n
                    logger.warning(
                        f"'{key}' is not in states. Inferring number of states "
                        f"from classified data."
                    )
            
            if n < default_n:
                logger.warning(
                    f"Getting state populations for {n} states but '{key}' has "
                    f"classified data for more than {n} states"
                )

            shape = tuple(n if axis == self.axis else dim for axis, dim in enumerate(classified.shape))
            outputs[key] = np.empty(shape=shape)

            for i in range(n):
                outputs[key][i] = np.mean(classified == i, axis=0)

        return outputs