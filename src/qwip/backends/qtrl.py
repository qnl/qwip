import re

import numpy as np
from attrs import field
from numpy.typing import NDArray
from scipy.fft import fft, fftfreq, fftshift

try:
    from qtrl.managers import MetaManager
except ImportError:
    ...


from qwip.attrs import qdefine
from qwip.backends.backend import QuantumBackend
from qwip.processing.processors import IQResult, array_real_to_complex
from qwip.qpu.systems import ReadoutResonator
from qwip.sequencer.compilation import (
    DeviceInfo,
    IntermediateProgram,
    PlayInstruction,
    QuantumExecutable,
    QWiPCompiler,
    WaitTriggerInstruction,
    register_compiler,
)
from qwip.sequencer.elements import SequenceElement
from qwip.sequencer.sequence import Sequence


@qdefine(kw_only=False)
class _ReadoutInfo:
    """Readout info.

    This is a mirror of the _ReadoutInfo class in `qtrl.sequence_utils.readout`.

    Attributes:
        sequence: A WaveformData object to mirror a QTRL sequence.
        qubits: A list of qubit indices corresponding to which qubits a read out.
        n_readouts: The total number of readouts across all sequence elements.
    """

    sequence: "SequenceArray"
    qubits: list[int]
    n_readouts: int


@qdefine
class SequenceArray:
    sample_rate: float
    n_elements: int
    num_channels: int
    array: NDArray[np.float32] | None = None
    x_axis: np.ndarray | None = None
    is_array_compiled: bool = True
    readout_locations: dict = field(factory=dict)
    _readout: _ReadoutInfo | None = None

    def get_readout_locations(self):
        return self.readout_locations

    def get_truncations(self):
        return self.readout_locations

    @property
    def shape(self) -> tuple[int, ...]:
        return self.array.shape

    def generate_seq_table(self, elem_len=None):
        """Copied from the old sequencer

        Split a 4 dimensional matrix into a list of unique elements and a sequence table from which to
        recreate the original.
            Input:
                elem_len: - Length of which to chop up the waveforms into smaller pieces in an attempt
                                   to find the list of unique elements.

            Returns:
                - Unique Waveforms - an array containing the unique chunks of all the waveforms, of dimension 3,
                                    Dimension 0 - the number of unique elements found
                                    Dimension 1 - the number of output elements, IE analog, mk1, mk2 etc
                                    Dimension 2 - elem_len and contains the actual waveform chunk.

                - seq_table - is what will become the sequence table.  It is laid out as so:
                                Dimension 0 - Channel of the AWG
                                Dimension 1 - Sequence Element Number
                                Dimension 2 - Chunk Number (this is waveform length/ elem_len)
                                The integer value n listed in the chunk number corresponds to the nth unique element
                                from the unique waveform array above.

            The original waveform can then be reconstructed in full using the results with the command:

                unique_waveforms[seq_table].reshape(num_chans, seq_len, elem_len*wav_len, -1)
        """
        seq_array = self.array
        if elem_len is None:
            elem_len = seq_array.shape[-2]

        if len(seq_array.shape) != 4:
            raise Exception(
                "Matrix must be of dimension 4, [0] is channels, [1] is sequence number, [2] is waveform, [3] is outport"
            )

        if seq_array.shape[2] % elem_len != 0:
            raise Exception("Waveform length is not a multiple of element length")

        num_chans = seq_array.shape[0]
        seq_len = seq_array.shape[1]
        num_outputs = seq_array.shape[3]
        wav_len = int(seq_array.shape[2] / elem_len)

        seq_array = np.round(seq_array, 5)

        # Make what will become the sequence table
        seq_table = np.zeros(
            seq_array.reshape(num_chans, seq_len, -1, elem_len, num_outputs).shape[0:3]
        )

        # Chunk the waveform table into something of the right shape
        chunked_waveforms = seq_array.reshape(
            num_chans, seq_len, -1, elem_len, num_outputs
        )

        # Hash the waveforms and stuff the hashes into the sequence table, we will replace these with more sensical
        # numbers later
        for chan in range(seq_table.shape[0]):
            for seq_elem in range(seq_table.shape[1]):
                for wav_elem in range(seq_table.shape[2]):
                    seq_table[chan, seq_elem, wav_elem] = hash(
                        chunked_waveforms[chan, seq_elem, wav_elem].tostring()
                    )

        # Now we can count our unique hashes!
        unique_hashes = np.unique(seq_table)
        # print("Found {} unique chunks".format(len(unique_hashes)))
        # ok, lets relabel our hashed values to a more normal numbering from 0- unique number of hashes
        # lets make a little dictionary for this
        relabeling = dict(zip(unique_hashes, np.arange(unique_hashes.shape[0])))

        # change our seq_table to reflect this new numbering
        for chan in range(seq_table.shape[0]):
            for seq_elem in range(seq_table.shape[1]):
                for wav_elem in range(seq_table.shape[2]):
                    seq_table[chan, seq_elem, wav_elem] = relabeling[
                        seq_table[chan, seq_elem, wav_elem]
                    ]
        seq_table = seq_table.astype(int)

        # now we need to collect our unique waveforms, we can use our spiffy new seq_table for that
        # here is the array we will stuff these into
        unique_waveforms = np.zeros((unique_hashes.shape[0], elem_len, num_outputs))

        # ok, we can use np.where to get a list of where all the elements are, take the first location
        # and use that location information to get the waveform element from the chunked array,
        # then stuff that into our unique_waveform array
        for i in range(unique_hashes.shape[0]):
            loc = np.where(seq_table == i)
            unique_waveforms[i] = chunked_waveforms[loc[0][0], loc[1][0], loc[2][0]]

        # Verify we built what we wanted
        if (
            np.sum(
                unique_waveforms[seq_table].reshape(
                    num_chans, seq_len, elem_len * wav_len, -1
                )
                != seq_array
            )
            != 0
        ):
            raise Exception(
                "Sequence chunking failed, might be a hash collision (unlikely)."
            )

        # Huzzah, we have a nice sequence table and list of unique elements
        return unique_waveforms, seq_table

    def fft(
        self, include_freqs: bool = True
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Computes the fourier transform of the waveform data.

        Args:
            include_freqs: If true, also returns the frequency array.

        Returns:
            Either a tuple containing frequency array and the fft data or just
            the fft data.
        """
        ys = fftshift(fft(self.array, axis=2))

        if include_freqs:
            xs = fftshift(fftfreq(self.shape[2], 1 / self.sample_rate))
            return xs, ys

        return ys


@qdefine
class QTRLExecutable(QuantumExecutable):
    """Compiled sequence.

    Compiled sequences should be specific to the hardware it is meant to be run
    on. This is meant to plug into existing qtrl code.
    """

    waveforms: dict[str, SequenceArray] = field(factory=dict)

    @property
    def array(self) -> np.ndarray:
        return self.waveforms["seq"].array

    @property
    def _readout(self) -> SequenceArray:
        return self.waveforms["readout"]

    @property
    def is_array_compiled(self) -> bool:
        return self.waveforms["seq"].is_array_compiled

    @property
    def shape(self) -> tuple[int, ...]:
        return self.waveforms["seq"].shape

    @property
    def n_elements(self) -> int:
        return self.waveforms["seq"].n_elements

    @property
    def x_axis(self) -> np.ndarray:
        return np.arange(self.array.shape[1])

    def get_readout_locations(self) -> dict[int, int]:
        return self.waveforms["seq"].get_readout_locations()

    def get_truncations(self) -> dict[int, int]:
        return self.waveforms["seq"].get_readout_locations()

    def generate_seq_table(self, elem_len=None):
        return self.waveforms["seq"].generate_seq_table(elem_len=elem_len)

    def fft(self) -> dict[str, np.ndarray]:
        """Computes the fourier transform of the CompiledSequence

        Returns:
            A dictionary mapping sequence groups to their frequency domain
            representation.
        """
        fftdict = {
            key: wavedata.fft(include_freqs=False)
            for key, wavedata in self.waveforms.items()
        }

        return fftdict


def populate_unpaired(exe: QTRLExecutable, fill: float = 1 / 2**15) -> None:
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
    has_wave = np.any(exe.array, axis=2)[..., 0]  # ignore marker array

    for ch1, ch2 in np.arange(exe.shape[0]).reshape(-1, 2):
        # bitwise xor to find all unpaired channels
        unpaired = has_wave[ch1] ^ has_wave[ch2]

        exe.array[ch1, unpaired & ~has_wave[ch1], 0, 0] = fill
        exe.array[ch2, unpaired & ~has_wave[ch2], 0, 0] = fill


def format_legacy_IQ(arr: np.ndarray) -> np.ndarray:
    """Reformats a QTRL result array.

    This function will reorder the axis so that the IQ data for each shot is
    contiguous. The legacy heterodyne array is a 4-D array where the axes correspond
    to `(IQ, shots, elements, readouts)`. This is reformatted to a complex numpy array
    where the shape is `(shots, elements, readouts)`.
    """

    return array_real_to_complex(np.transpose(arr, [1, 2, 3, 0]))[..., 0]


@register_compiler
@qdefine
class QTRLCompiler(QWiPCompiler):
    def compile(
        self, seq: Sequence, location_kwargs: dict = {}, pulse_kwargs: dict = {}
    ) -> QTRLExecutable:
        qwip_exe = super().compile(seq, location_kwargs, pulse_kwargs)

        readout_qubits = set()

        waveforms = {}
        for dev_name, program in qwip_exe.programs.items():
            readout_qubits.update(program.read_registers)
            if program.waveforms:
                waveforms[dev_name] = self.compile_sequence_array(program)

        trigger = self.channels["readout"].trigger
        marker = (trigger.index, trigger.subchannel)
        waveforms["seq"].readout_locations = {
            el: sample
            for el, sample in enumerate(qwip_exe.programs["seq"].markers[marker])
        }

        try:
            waveforms["readout"]._readout = _ReadoutInfo(
                sequence=waveforms["readout"],
                qubits=list(readout_qubits),
                n_readouts=sum(qwip_exe.num_reads),
            )
        except KeyError:
            pass

        return QTRLExecutable(sequence=seq, waveforms=waveforms)

    def compile_sequence_array(self, program: IntermediateProgram) -> SequenceArray:
        device = self.channels[program.device]
        n_channels = device.max_channel_index + 1
        n_elements = len(program.waveforms)
        n_samples = max(wmem.samples for wmem in program.waveforms)
        n_subchannels = device.max_subchannel_index + 1

        seq_arr = SequenceArray(
            sample_rate=device.sample_rate,
            n_elements=n_elements,
            num_channels=n_channels,
            array=np.zeros(
                (n_channels, n_elements, n_samples, n_subchannels), dtype=np.float32
            ),
        )

        for el, wmem in enumerate(program.waveforms):
            for (ch, subch), wavedata in wmem.data.items():
                seq_arr.array[ch, el, : len(wavedata), subch] = wavedata

        return seq_arr


@qdefine
class QTRLBackend(QuantumBackend):
    """A hardware backend that interface with QTRL."""

    meta: "MetaManager"

    def upload(self, exe: QTRLExecutable, **kwargs) -> None:
        populate_unpaired(exe)
        self.uploaded = exe
        self.meta.write_sequence(exe)

    def acquire(self, exe: QTRLExecutable, repetitions: int = 512, **kwargs) -> dict:
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

        for r in qpu.compiler.readout_qubits:
            pulse_name = readout_config.drives[f"R{r}"]

            ro_se += qpu.db.load_pulse(pulse_name, {length_variable: length})

        return ro_se

    def exe_formats(self) -> set[type[QuantumExecutable]]:
        return {QTRLExecutable}
