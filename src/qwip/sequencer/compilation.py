import itertools as it
from abc import ABCMeta
from collections.abc import Callable, Collection, Iterable
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from attrs import evolve, field
from loguru import logger
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray
from scipy.fft import fft, fftfreq, fftshift
from typing_extensions import Self

from qwip.attrs import qdefine, qfrozen
from qwip.sequencer.elements import SequenceElement
from qwip.sequencer.phase_tracker import ModulationFrequency, PhaseTracker, PhaseUpdater
from qwip.sequencer.sequence import Sequence
from qwip.sequencer.utils import Location
from qwip.sequencer.waveform import Marker, ReadoutMarker, TriggeredWaveform, Waveform
from qwip.visualization.utils import all_legend_handles_labels

REGISTERED_SEQUENCERS: dict[str, "WaveformSequencer"] = dict()


def register_sequencer(cls: type["WaveformSequencer"]) -> type["WaveformSequencer"]:
    if not issubclass(cls, WaveformSequencer):
        raise TypeError(f"Registered sequencer must subclass {WaveformSequencer}")

    REGISTERED_SEQUENCERS[cls.__name__] = cls

    return cls


def find_end_marker(locations, name="end") -> Location | None:
    for loc, waves in locations.items():
        if Marker(name=name) in waves:
            return loc

    return None


@qdefine
class QuantumExecutable(metaclass=ABCMeta):
    """An abstract base class for hardware-specific executables."""

    sequence: Sequence | None = field(eq=id, default=None)

    @property
    def seq(self) -> Sequence | None:
        return self.sequence


def find_readout_marker(locations) -> Location | None:
    readout_location = None
    for loc, waves in locations.items():
        for w in waves:
            if not isinstance(w, ReadoutMarker):
                continue

            if readout_location is not None:
                raise ValueError(
                    "Only one terminal readout marker per sequence element is supported."
                )

            readout_location = loc

    return readout_location


@qdefine(kw_only=False)
class _ReadoutInfo:
    """Readout info.

    This is a mirror of the _ReadoutInfo class in `qtrl.sequence_utils.readout`.

    Attributes:
        sequence: A WaveformData object to mirror a QTRL sequence.
        qubits: A list of qubit indices corresponding to which qubits a read out.
        n_readouts: The total number of readouts accross all sequence elements.
    """

    sequence: "WaveformData"
    qubits: list[int]
    n_readouts: int


@qdefine
class WaveformData:
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


@qfrozen(kw_only=False)
class ChannelInfo:
    """Holds information mapping a logical channel name to a physical channel index.

    Attributes:
        name: The logical channel name.
        index: The physical channel index (0-indexed) corresponding to the hardware channel.
        group: The name of the channel group this channel belongs to.
        subchannel: The subchannel (used for markers) that this channel name refers to.
        read: True if the channel is an ADC channel.
        delay: A channel delay in ns to add to all waves on this channel.
    """

    name: str
    index: int
    group: str | None = None
    subchannel: int = 0  # Use nonzero for markers
    read: bool = False
    delay: float = 0


@qfrozen
class TriggerInfo:
    device: str
    index: int
    subchannel: int = 0


@qfrozen
class ChannelGroup:
    """Holds information about a group of logical channels.

    A channel group typically corresponds to a single hardware box or set of boxes that are
    addressed together. They should all share the same sample rate.

    Attributes:
        name: The name of the channel group.
        channels: A tuple of all channels that belong to this group.
        sample_rate: The sampling rate of the channel group in samples/second.
    """

    name: str
    channels: tuple[ChannelInfo, ...] = field(
        repr=lambda channels: repr(tuple(ch.name for ch in channels))
    )
    sample_rate: float
    trigger: TriggerInfo | None = None

    @property
    def num_channels(self) -> int:
        return len(self.channels)

    @property
    def num_subchannels(self) -> int:
        return len(set(ch.subchannel for ch in self))

    @property
    def max_channel_index(self) -> int:
        return max(ch.index for ch in self)

    @property
    def max_subchannel_index(self) -> int:
        return max(ch.subchannel for ch in self)

    def get_max_channel_delay(self) -> int:
        """Returns the largest channel delay specified within the channel group."""
        return max(ch.delay for ch in self)

    def channel_names(self) -> set[str]:
        return {ch.name for ch in self.channels}

    def channel_indices(self) -> set[int]:
        return {ch.index for ch in self.channels}

    def has_output(self) -> bool:
        for ch in self.channels:
            if not ch.read:
                return True
        return False

    @classmethod
    def from_channels(
        self,
        channels: Iterable[ChannelInfo],
        sample_rate: float,
        name: str | None = None,
        **kwargs,
    ) -> Self:
        """Constructs a channel group from a list of channels.

        Args:
            channels: The channels that belong to the group. They must all have unique names,
                and should not have conflicting groups. If the group attribute for all the
                channels is None, then name must be provided.
            sample_rate: The sampling rate for the channel group.
            name: The name of the channel group. This can be inferred from the channels if
                they specify a group. Otherwise, name must be provided.

        Returns:
            The channel group.
        """
        group = name
        names = set()

        for ch in channels:
            if group and ch.group and ch.group != group:
                raise ValueError(
                    "Cannot create channel group from channels with different group names."
                )

            group = group or ch.group

            if ch.name in names:
                raise ValueError("Channels must all have unique names!")

            names.add(ch.name)

        channels = tuple(ch if ch.group else evolve(ch, group=group) for ch in channels)

        return ChannelGroup(
            name=group, channels=channels, sample_rate=sample_rate, **kwargs
        )

    def __iter__(self):
        yield from self.channels.__iter__()

    def __getitem__(self, channel_name: str):
        try:
            return next(ch for ch in self if ch.name == channel_name)
        except StopIteration as e:
            raise KeyError(f"'{channel_name}'") from e


@qdefine
class CompiledSequence(QuantumExecutable):
    """Compiled sequence.

    Compiled sequences should be specific to the hardware it is meant to be run
    on. This is meant to plug into existing qtrl code.
    """

    waveforms: dict[str, WaveformData] = field(factory=dict)

    @property
    def array(self) -> np.ndarray:
        return self.waveforms["seq"].array

    @property
    def _readout(self) -> WaveformData:
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

    def plot(
        self,
        element: int | None = None,
        title: str = "Pulse Sequence Simulation",
        channels: list[tuple[int, ...]] | None = None,
        axes: Collection[Axes] | None = None,
        fig_props: dict = {},
    ) -> Figure:
        plotter = CompiledSequencePlotter()

        return plotter.plot(
            self, element, title, channels, axes, fig_props
        )  # plotter.plot(self, element, channels, axes, fig_props)


@qdefine
class WaveformSequencer:
    """A waveform sequencer.

    The waveform sequencer is responsible for compiling sequences into concrete timepoints
    that can then be uploaded to the measurement hardware (DAC/ADC).

    Attributes:
        channels: A mapping from channel group names to `ChannelGroup` instances that
            contain information about the channels.
        modulations: A mapping from modulation keys for phase tracking to concrete
            modulation frequencies.
        readout_qubits: The list of qubits that should be included in the hardware
            demodulation weights that are uploaded to the ADC. This is a legacy
            parameter necessary for the ZI UHFQA's.
        end_marker: A string specifying the marker name that is used to specify the
            end of a sequence element.
    """

    channels: dict[str, ChannelGroup] = field(factory=dict)
    modulations: dict[str, ModulationFrequency] = field(factory=dict)
    readout_qubits: list[int] = field(factory=list)
    end_marker: str = "end"

    @classmethod
    def from_channel_groups(
        cls, channel_groups: Iterable[ChannelGroup], **kwargs: Any
    ) -> Self:
        """Contruct the waveform sequencer from a list of `ChannelGroup`.

        Args:
            channel_groups: A list of `ChannelGroup` instances representing the
                measurement hardware.
            **kwargs: Remaining keyworad arguments are passed to the `__init__`
                function.

        Returns:
            A new WaveformSequencer instance.
        """
        channels = {ch_group.name: ch_group for ch_group in channel_groups}

        return cls(channels=channels, **kwargs)

    def get_channel_info(self, name: str) -> ChannelInfo | None:
        """Returns the `ChannelInfo` with the given name.

        It is assumed that there are no repeated channel names between channel groups,
        so this method will short circuit on the first channel that matches the name.

        Args:
            name: The name of the channel to get.

        Returns:
            A `ChannelInfo` or `None`, if no channel matching the name exists.
        """
        for group in self.channels.values():
            try:
                return group[name]
            except KeyError:
                continue

        return None

    def compile_phases(self, locations: dict[Location, list[Waveform]]) -> PhaseTracker:
        """Returns a new phase tracker instance with all virtual phase updates.

        Every waveform that has an `update_phase_tracker` method will be called
        on the `PhaseTracker`.

        Args:
            locations: A dictionary mapping locations to waveforms.

        Returns:
            An updated phase tracker.
        """
        phase_tracker = PhaseTracker.from_modulations(self.modulations)

        for loc, waves in locations.items():
            loc = loc.offset

            for w in waves:
                if not isinstance(w, PhaseUpdater):
                    continue

                w.update_phase_tracker(loc, phase_tracker)

        return phase_tracker

    def compile_timepoints(
        self,
        locations: dict[Location, list[Waveform]],
        waveform_array: NDArray[np.float32],
        channel_group: ChannelGroup,
        phase_tracker: PhaseTracker,
        pulse_kwargs: dict = {},
    ) -> None:
        """Compiles a single timeline of pulses into concrete timepoints.

        Each element represents a DAC amplitude (normalized between -1 and 1)
        on a specific channel/subchannel for a given sample timestep.

        Args:
            locations: A dictionary mapping locations to waveforms. The locations
                should be time ordered.
            waveform_array: A numpy array with shape `(channels, timepoints, subchannels)`
                that will hold the compiled timepoints
            channel_group: The channel group that corresponds to this location
                map.
            phase_tracker: A phase tracker instance that holds all phase jumps for
                this timeline of pulses.
            pulse_kwargs: A mapping of variable names to resolved values to pass to
                all pulses.
        """
        sample_rate = channel_group.sample_rate
        num_timepoints = waveform_array.shape[1]

        ts = np.arange(num_timepoints) / sample_rate
        logger.debug(f"Time array shape: {ts.shape}.")

        for loc, waves in locations.items():
            for w in waves:
                width = w.width.resolve(**pulse_kwargs)
                start, end = loc.offset, loc.offset + width.offset

                s_idx = int(start * sample_rate)
                e_idx = num_timepoints if np.isinf(end) else int(end * sample_rate) + 1
                if s_idx == e_idx - 1:
                    continue

                ts_wave = ts[s_idx:e_idx]

                w_t = w(
                    ts_wave,
                    t0=start + w.t0,
                    phase_tracker=phase_tracker,
                    modulations=self.modulations,
                    **pulse_kwargs,
                )

                if len(w_t.shape) == 1:
                    w_t = w_t[np.newaxis, :]

                for i, c in enumerate(w.channels):
                    ch_idx = (channel_group[c].index,)
                    subchannel = channel_group[c].subchannel
                    waveform_array[ch_idx, s_idx:e_idx, subchannel] += w_t[i]

        return waveform_array

    def compile_sequence_element(
        self,
        locations: dict[Location, list[Waveform]],
        waveform_array: np.ndarray,
        channel_group: ChannelGroup,
        pulse_kwargs: dict = {},
    ):
        """Compiles a single sequence elements.

        This method makes two passes through location waveform mapping. The
        first pass compiles all phase jumps and the second pass evaluates the
        pulse timepoints.

        Args:
            locations: A dictionary mapping locations to waveforms. The locations
                should be time ordered.
            waveform_array: A numpy array with shape `(channels, timepoints, subchannels)`
                that will hold the compiled timepoints
            channel_group: The channel group that corresponds to this location
                map.
            pulse_kwargs: A mapping of variable names to resolved values to pass to
                all pulses.

        """
        # Compile phases
        phase_tracker = self.compile_phases(locations)

        return self.compile_timepoints(
            locations=locations,
            waveform_array=waveform_array,
            channel_group=channel_group,
            phase_tracker=phase_tracker,
            pulse_kwargs=pulse_kwargs,
        )

    def initialize_compiled_sequence(
        self, seq: Sequence, max_times: dict[str, float]
    ) -> CompiledSequence:
        """Creates a new `CompiledSequence` instance.

        This function allocates the waveform arrays that will hold all the waveform
        data for the given sequence.

        Args:
            seq: The sequence to be compiled.
            max_times: A dictionary mapping channel group keys to the latest timepoint
                played on any channel accross all sequence elements. Times are specified
                in seconds.

        Returns:
            A new `CompiledSequence` instance with the waveform data arrays initialized
            to all zeros.
        """

        waveform_arrs = dict()
        for key, group in self.channels.items():
            sample_rate = group.sample_rate
            num_channels = group.max_channel_index + 1
            num_subchannels = group.max_subchannel_index + 1

            num_elements = np.prod(seq.shape) if key == "seq" else 1
            num_timepoints = int(max_times[key].offset * sample_rate)

            # We put num_subchannels as the first index and then transpose in an
            # attempt to make memory layout more sensible.
            arr_shape = (num_subchannels, num_channels, num_elements, num_timepoints)
            logger.debug(f"Channel group {key} shape: {arr_shape}.")

            waveform_arrs[key] = WaveformData(
                sample_rate=sample_rate,
                n_elements=num_elements,
                num_channels=num_channels,
                # This should return a view of the array
                array=np.zeros(arr_shape, dtype=np.float32).transpose(1, 2, 3, 0),
            )
            logger.debug(f"dtype: {waveform_arrs[key].array.dtype}")

        return CompiledSequence(waveforms=waveform_arrs, sequence=seq)

    def compile(
        self,
        seq: Sequence,
        location_kwargs: dict = {},
        pulse_kwargs: dict = {},
        **triggered_elements,
    ) -> CompiledSequence:
        """Compiles a sequence.

        This function takes an abstract sequence and compiles it into a concrete
        set of timepoints.

        Args:
            seq: The sequence to compile.
            location_kwargs: Any location constraints to add to the sequence
                before compilation.
            pulse_kwargs: A mapping of variables names to resolved pulse parameters.

        Returns:
            A `CompiledSequence` instance.
        """
        # First resolve all locations in the main sequence
        locations = [
            se.resolve_locations(end_marker=self.end_marker, **location_kwargs)
            for se in seq.flat
        ]

        # Then for any triggered sequence elements (most commonly readout)
        triggered_locations = {
            k: se.resolve_locations(end_marker=self.end_marker, **location_kwargs)
            for k, se in triggered_elements.items()
        }

        # The above step was necessary to determine the number of timepoints
        # in the waveform array
        max_times = dict(seq=0) | {
            k: find_end_marker(se_locs, self.end_marker)
            for k, se_locs in triggered_locations.items()
        }

        max_times["seq"] = max(
            find_end_marker(se_locs, self.end_marker) for se_locs in locations
        )
        logger.debug(max_times)

        # Then use the collected information to initialize an empty CompiledSequence
        # with the correct sizes for all waveform data arrays
        cseq = self.initialize_compiled_sequence(seq, max_times)

        # Now we move on to actually compiling timepoints and writing them to the
        # waveform data arrays
        for i, (se_locs, se) in enumerate(zip(locations, seq.flat)):
            logger.debug(cseq.waveforms["seq"].sample_rate)
            logger.debug(se.constraints | pulse_kwargs)
            self.compile_sequence_element(
                se_locs,
                # (channel_idx, element_idx, timepoints, num_outports)
                cseq.waveforms["seq"].array[:, i, :],
                self.channels["seq"],
                se.constraints | pulse_kwargs,
            )

        # Then do the same for triggered sequence_elements
        for trigger, se_locs in triggered_locations.items():
            se = triggered_elements[trigger]
            self.compile_sequence_element(
                se_locs,
                # (channel_idx, element_idx, timepoints, num_outports)
                cseq.waveforms[trigger].array[:, 0, :],
                self.channels[trigger],
                se.constraints | pulse_kwargs,
            )

        # Finally we pull out all the readout locations in the sequence
        for i, se_locs in enumerate(locations):
            sample_rate = cseq.waveforms["seq"].sample_rate
            rloc = int(find_readout_marker(se_locs).offset * sample_rate)
            cseq.get_readout_locations()[i] = rloc

        # This is needed for backwards compatibility.
        rinfo = _ReadoutInfo(
            cseq._readout, self.readout_qubits, len(cseq.get_readout_locations())
        )
        cseq._readout._readout = rinfo

        return cseq


register_sequencer(WaveformSequencer)


@qdefine
class CompiledSequencePlotter:
    axsize: tuple[float, float] = (8, 1)

    def make_axes(
        self,
        n: int,
        axsize: tuple[float, float] | None = None,
        sharex: bool = True,
        sharey: bool = True,
        **props,
    ) -> Figure:
        """Creates a matplotlib figure and axes.

        Args:
            n: Number of axes.
            axsize: The size (width, height) in inc
        """
        if "figsize" not in props:
            axsize = axsize or self.axsize
            width, height = axsize

            if width == height == ...:
                width, height = (8, 1)
            elif width is ...:
                width = 8 / height
            elif height is ...:
                height = 1 / 8 * width

            props["figsize"] = (width, n * height)

        fig, _ = plt.subplots(n, 1, sharex=sharex, sharey=sharey, **props)

        return fig

    def plot(
        self,
        cseq: CompiledSequence,
        element: int,
        title: str,
        channels: list[tuple[int, ...]] | None = None,
        axes: Collection[Axes] | None = None,
        fig_props: dict = {},
    ) -> Figure:
        tdict = {
            name: np.arange(waveformdata.array.shape[2]) / waveformdata.sample_rate
            for name, waveformdata in cseq.waveforms.items()
        }

        mainseq = cseq.waveforms["seq"]
        ts = tdict["seq"]

        if channels is None:
            channels = [tuple(ch for ch in range(mainseq.array.shape[0]))]

        if axes is None:
            fig = self.make_axes(len(channels) + 1, **fig_props)
            axes = fig.axes

            if isinstance(axes, Axes):
                axes = np.array([axes])

        for ax_id, ch_group in enumerate(channels):
            ax = axes[ax_id]

            for ch in ch_group:
                for marker in range(mainseq.array.shape[3]):
                    pts = mainseq.array[ch, element, :, marker]
                    if not pts.any():
                        continue

                    ax.plot(ts, pts, label=f"CH{ch}", color=f"C{ch}")

        readoutseq = cseq.waveforms["readout"]

        treadout = mainseq.get_readout_locations()[element] / mainseq.sample_rate
        for ch in range(readoutseq.array.shape[0]):
            pts = readoutseq.array[ch, 0, :, 0]
            props = dict(color=f"C{ch + mainseq.shape[0]}", label=f"Readout CH{ch}")
            axes[-1].plot(tdict["readout"] + treadout, pts, **props)

        figwidth, _ = fig.get_size_inches()

        h, l = all_legend_handles_labels(axes)
        axes[0].legend(
            h,
            l,
            mode="expand",
            bbox_to_anchor=(0, 1.05, 1, 0.05),
            loc="lower left",
            ncols=min(figwidth // 2, len(l)),
            borderaxespad=0,
        )
        axes[0].set_ylim(-1, 1)

        axes[-1].set_xlabel("Time (s)")
        return fig


@qdefine
class InteractiveSequencePlotter:
    def plot(
        self,
        cseq: CompiledSequence,
        title: str = "Pulse Sequence Simulation",
        channels: list[tuple[int, ...]] | None = None,
        axes: Collection[Axes] | None = None,
        fig_props: dict = {},
    ) -> Figure:
        fig = go.Figure()

        ts_pulse = (
            np.arange(cseq.waveforms["seq"].array.shape[2])
            / cseq.waveforms["seq"].sample_rate
        )
        N_channels, N_elements, N_steps, N_subchannels = cseq.waveforms[
            "seq"
        ].array.shape

        active_elements = dict()
        num_traces = 0

        for i in range(N_elements):
            active_channels = np.where(
                np.any(
                    cseq.waveforms["seq"].array[:, i, :, 0].reshape(N_channels, -1),
                    axis=1,
                )
            )[0]

            if len(active_channels) != 0:
                active_elements[i] = active_channels

            for ch in active_channels:
                fig.add_trace(
                    go.Scatter(
                        x=ts_pulse,
                        y=cseq.waveforms["seq"].array[ch, i, :, 0],
                        visible=False,
                        name=f"CH {ch}",
                    )
                )
                num_traces += 1

        # Slider to filter Sequence Element
        steps_seq, start = [], 0
        last_element = 0

        for index, (element, targets) in enumerate(active_elements.items()):
            visible_seq = [False] * num_traces
            end = start + len(targets)

            visible_seq[start:end] = [True] * (end - start)
            steps_seq.append(
                dict(
                    label=f"{element}", method="update", args=[{"visible": visible_seq}]
                )
            )
            if index == len(active_elements.items()) - 1:
                for i in range(start, end):
                    fig.data[i].visible = True
                last_element = index

            start = end

        sliders = [
            dict(
                active=last_element,
                currentvalue={"prefix": "Sequence Element: "},
                pad={"t": 50, "b": 50},
                steps=steps_seq,
                borderwidth=2,
            )
        ]

        fig.update_layout(
            sliders=sliders,
            dragmode="pan",
            title={"text": title, "x": 0.5, "xanchor": "center"},
            yaxis_title="Amplitude",
            xaxis_title="Time",
            width=1000,
            height=600,
            autosize=False,
            margin=dict(t=50, b=0, l=0, r=0),
        )

        fig.update_yaxes(fixedrange=True)

        config = {"scrollZoom": True}
        fig.show(config=config)

        return fig


__all__ = [
    "ChannelInfo",
    "ChannelGroup",
    "CompiledSequence",
    "CompiledSequencePlotter",
    "InteractiveSequencePlotter",
    "WaveformData",
    "WaveformSequencer",
]
