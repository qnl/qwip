<<<<<<< HEAD
=======
import re

>>>>>>> main
import numpy as np
from attrs import field

try:
    from qtrl.managers import MetaManager
except ImportError:
    ...


from qwip.attrs import qdefine
from qwip.backends.backend import QuantumBackend
<<<<<<< HEAD
from qwip.processing.processors import IQResult
=======
from qwip.processing.processors import IQResult, array_real_to_complex
>>>>>>> main
from qwip.qpu.systems import ReadoutResonator
from qwip.sequencer.compilation import CompiledSequence, QuantumExecutable
from qwip.sequencer.elements import SequenceElement


def populate_unpaired(cseq: CompiledSequence, fill: float = 1 / 2**15) -> None:
    """Ensures that waveform data is paired.

    Currently there is still a bug when uploading waves to a ZI HDAWG where waves are
    uploaded incorrectly unless they are paired. This function modifies a compiled
    qtrl sequence so that all channels with waveform data are paired.

    Args:
        cseq: The compiled sequence to modify.
        fill: The DAC amplitude to set on an empty unpaired channel. This should be
            small to avoid any adverse affect on the system. This value is added to the
            first sample only.
    """
    has_wave = np.any(cseq.array, axis=2)[..., 0]  # ignore marker array

    for ch1, ch2 in np.arange(cseq.shape[0]).reshape(-1, 2):
        # bitwise xor to find all unpaired channels
        unpaired = has_wave[ch1] ^ has_wave[ch2]

        cseq.array[ch1, unpaired & ~has_wave[ch1], 0, 0] = fill
        cseq.array[ch2, unpaired & ~has_wave[ch2], 0, 0] = fill


def format_legacy_IQ(arr: np.ndarray) -> np.ndarray:
    """Reformats a QTRL result array.

    This function will reorder the axis so that the IQ data for each shot is
    contiguous. The legacy heterodyne array is a 4-D array where the axes correspond
    to `(IQ, shots, elements, readouts)`. This is reformatted to a complex numpy array
<<<<<<< HEAD
    where the shape is `(elements, readouts, shots)`.
    """
    match arr.dtype:
        case np.float32:
            cast = np.complex64
        case np.float64:
            cast = np.complex128
        case _:
            arr = arr.astype(float)
            cast = np.complex128

    return np.array(np.transpose(arr, [2, 3, 1, 0]), order="C").view(cast)[..., 0]
=======
    where the shape is `(elements, shots, readouts)`.
    """

    return array_real_to_complex(np.transpose(arr, [2, 1, 3, 0]))[..., 0]
>>>>>>> main


@qdefine
class QTRLBackend(QuantumBackend):
    """A hardware backend that interface with QTRL."""

    meta: "MetaManager"
    ro_se: SequenceElement = field(factory=SequenceElement)

    def upload(self, exe: CompiledSequence, **kwargs) -> None:
        populate_unpaired(exe)
        self.meta.write_sequence(exe)

    def acquire(self, exe: CompiledSequence, repetitions: int = 512, **kwargs) -> dict:
        acquisition_kwargs = dict(n_reps=repetitions, save_data=False) | kwargs

        meas = self.meta.acquire(**acquisition_kwargs)
        iqdata = {}

        for k in meas:
            if not re.match(r"R(\d+)", k):
                continue

            IQ = format_legacy_IQ(meas[k]["Heterodyne"])

            iqdata[k] = IQResult.from_numpy(IQ, name=k)

        return iqdata

    def update_parameters(
        self, qpu: "QPU", readout: dict | SequenceElement = {}, **kwargs
    ):
        """Updates parameters from the QPU.

        The QTRL backend requires that all readout frequencies are specified in the
        variables config because this is where the ADCManager pulls the demod
        frequencies from.
        """
        for sys in qpu.subsystems.values():
            match sys:
                case ReadoutResonator(name=n, frequency=f):
                    if not (m := re.match(r"R(\d+)", n)):
                        raise ValueError(
                            f"Resonator names must be of the form 'R\\d+' for the "
                            f"QTRL backend. Got {n}"
                        )

                    self.meta.variables[f"Q{m[1]}/res_freq"] = f

        match readout:
            case dict():
                ro_se = self.get_readout_sequence(qpu, **readout)
            case SequenceElement():
                ro_se = readout
            case _:
                raise ValueError(
                    f"Readout must be a sequence element or a dictionary of parameters. "
                    f"Got {readout}"
                )

        self.ro_se = ro_se

    def get_readout_sequence(
        self,
        qpu: "QPU",
        readout: str = "default",
        length: float | None = None,
        length_variable: str = "width",
    ) -> SequenceElement:
        """Constructs a readout sequence element from the readout config.

        Args:
            readout: The name of the readout config.
            length: The readout length in seconds.
            length_variable: The pulse variable that corresponds to the pulse width in
                the readout pulse.

        Returns:
            The readout sequence element.
        """
        readout_config = qpu.config.readout[readout]

        ro_se = SequenceElement()
        length = length or readout_config.length

        for r in qpu.sequencer.readout_qubits:
            pulse_name = readout_config.drives[f"R{r}"]

            ro_se += qpu.db.load_pulse(pulse_name, {length_variable: length})

        return ro_se

    def exe_formats(self) -> set[type[QuantumExecutable]]:
        return {CompiledSequence}
