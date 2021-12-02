import numpy as np

from numpy.random import default_rng

from loguru import logger

from qtrl.managers.ADC_manager import ADCManager
from qtrl.managers.DAC_manager import DACManager
from qtrl.settings import _Settings

class SimulatedADCManager(ADCManager):
    rng = default_rng()

    def _arm(self):
        self._num_shots = self.get('acquisition_settings/n_reps')

    def get_gmms(self):
        simple_readout = None
        for key in self['processing'].keys():
            if 'simple_readout' in key:
                simple_readout = key
        
        if not simple_readout:
            logger.error('Simple readout not found in processing')
        
        gmms = None
        for key in self['processing'][simple_readout].keys():
            if 'GMM' in key:
                gmms = key
        if not gmms:
            logger.error('GMM information not found in simple_readout')

        self._gmms = self['processing'][simple_readout][gmms]['kwargs']

    def generate_data(self, results, distributions=None):
        n_qubits, n_reps, n_readouts, n_triggers = results.shape
        num_states = {
            q: len(self._gmms['covariances'][f'R{q}']) for q in self._readout_qubits
        }

        dist_shape = (n_qubits, n_readouts, n_triggers, max(num_states.values()))
        
        if distributions is None:
            distributions = np.zeros(shape=dist_shape)
            for i, q in enumerate(self._readout_qubits):
                n = num_states[q]
                distributions[i, :, :, :n] = np.ones(n) / n
        else:
            assert distributions.shape == dist_shape

        assert np.all(np.isclose(distributions.sum(axis=3), 1))

        for q_idx, q in enumerate(self._readout_qubits):
            n = num_states[q]

            states = np.empty(shape=results.shape[1:])
            for element in range(n_readouts):
                for trig in range(n_triggers):
                    states[:,element,trig] = self.rng.choice(
                    n, size=n_reps, p=distributions[q_idx, element, trig]
                    )

            for i in range(n):
                sigma = self._gmms['covariances'][f'R{q}'][i]
                mean = self._gmms['means'][f'R{q}'][2*i:2*i+2]
                gmm = dict(mean=mean, cov=np.identity(n)*sigma)
                IQ = self.rng.multivariate_normal(
                    **gmm, size=np.count_nonzero(states == i)
                ).view(complex).squeeze()

                results[q_idx, states == i] = IQ

    def _acquire(self):
        self.get_gmms()
        shape = (
            len(self._readout_qubits),
            self._num_shots,
            self._num_readouts,
            2 if self._variables.get('readout/heralding', False) else 1
        )
        
        logger.debug(f'Results shape = {shape}')
        results = np.empty(shape, dtype=np.complex128)
        self.generate_data(results, distributions=self._settings.get('distributions'))
        return results

    def write_sequence(self, seq):
        readout = seq._readout
        self._readout_qubits = readout.qubits
        self._num_readouts = readout.n_readouts

class SimulatedDACManager(DACManager):
    def write_sequence(self, seq):
        return

    def run(self, state=1):
        return