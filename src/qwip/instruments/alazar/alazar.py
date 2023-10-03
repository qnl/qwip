"""Alazar device

!note "To add a new Alazar board"
    1. Create a new class like ATS9870 below
    2. Look up the AlazarTech user manual for your board (might be useful to
    compare with an existing board class: http://www.alazartech.com/support/Download%20Files/ATS9870%20User%20Manual_V1_2_Web.pdf)
    See atsapi.py for hex values, etc.
    3. Edit the dictionary values in your new class so they are consistent with
    your board's specifications
    4. Add your board to board_dict. Numerical value is given near line 158 of
    atsapi.py ('''Boards''')
    Now your board will automatically load according to the defaults and allowed
    values you've set. You're good to go!

    See Alazar SDK programmer's guide for more information:
    http://www.alazartech.com/support/Download%20Files/ATS-SDK-Guide-7.1.4.pdf
"""


import ctypes
import time

import numpy as np

from qwip.instruments.alazar import atsapi


class ATS9870(atsapi.Board):
    def __init__(self, **kwargs):
        self.name = "ATS9870"

        atsapi.Board.__init__(self, **kwargs)
        self.n_samples_per_chan, self.bits_per_sample = self.getChannelInfo()
        # self.n_samples_per_chan = int(self.n_samples_per_chan)
        ##Marie DEBUG 03-14-2020
        self.n_samples_per_chan = 10**6

        # self.bits_per_sample = int(self.bits_per_sample)
        self.bits_per_sample = 8

        self.bytes_per_sample = (self.bits_per_sample + 7) // 8

        self.record_limits = (128, 32)  # minimum length of record, 32 point step size

        self.channel_dict = {"CHAN_A": 1, "CHAN_B": 2}

        self.channel_impedance_dict = {50: 2}

        self.sample_rate_dict = {
            1e3: 0x1,  # Limited to 1 Gs/s. ATS9373 goes to 2 Gs/s
            2e3: 0x2,
            5e3: 0x5,
            10e3: 0x8,
            20e3: 0xA,
            50e3: 0xC,
            100e3: 0xE,
            200e3: 0x10,
            500e3: 0x12,
            1e6: 0x14,
            2e6: 0x18,
            5e6: 0x1A,
            10e6: 0x1C,
            20e6: 0x1E,
            50e6: 0x22,
            100e6: 0x24,
            200e6: 0x28,
            500e6: 0x30,
            1000e6: 0x35,
        }

        self.input_range_dict = {
            0.040: 0x2,
            0.100: 0x5,
            0.200: 0x6,
            0.400: 0x7,
            1: 0xA,
            2: 0xB,
            4: 0xC,
        }

        # actual allowable range 0 to Record Length-128
        self.max_pretrigger = 4096

        self.trigger_source_dict = {
            "CHAN_A": 0,
            "CHAN_B": 1,
            "EXTERNAL": 2,
            "DISABLE": 3,
        }

        self.trigger_slope_dict = {"POSITIVE": 1, "NEGATIVE": 2}

        # IMPORTANT: use names from atsapi.py. Update Alazar.check_clock_source if you add a new clock type
        self.clock_source_dict = {
            "INTERNAL": 0x1,
            "EXTERNAL": 0x2,
            "EXTERNAL_CLOCK_10MHz_REF": 0x7,
        }  # Sample rate must be 1e+9 for this clock source

        self.ext_trig_dict = {"ETR_1V": 1, "ETR_TTL": 2, "ETR_2V5": 3}

        self.channel_coupling_dict = {"DC": 2}

        # SET DEFAULTS
        self.channel_settings = {"CHAN_A": [0.4, "DC", 50], "CHAN_B": [0.4, "DC", 50]}

        self.trigger_settings = {
            "TRIG_1": ["DISABLE", "POSITIVE", 128],
            "TRIG_2": ["EXTERNAL", "POSITIVE", 128],
        }

        self.ext_trigger_settings = ["DC", "ETR_2V5"]

        self.clock_settings = ["EXTERNAL_CLOCK_10MHz_REF", 1e9, 0]


class ATS9360(atsapi.Board):
    def __init__(self, **kwargs):
        self.name = "ATS9360"

        atsapi.Board.__init__(self, **kwargs)
        self.n_samples_per_chan, self.bits_per_sample = self.getChannelInfo()
        self.n_samples_per_chan = self.n_samples_per_chan.value
        self.bits_per_sample = int(self.bits_per_sample.value)

        self.bytes_per_sample = (self.bits_per_sample + 7) // 8

        self.record_limits = (256, 128)  # minimum length of record, 128 point step size

        self.channel_dict = {"CHAN_A": 1, "CHAN_B": 2}

        self.channel_impedance_dict = {50: 2}

        self.sample_rate_dict = {
            1e3: 0x1,  # Limited to 1 Gs/s. ATS9373 goes to 2 Gs/s
            2e3: 0x2,
            5e3: 0x5,
            10e3: 0x8,
            20e3: 0xA,
            50e3: 0xC,
            100e3: 0xE,
            200e3: 0x10,
            500e3: 0x12,
            1e6: 0x14,
            2e6: 0x18,
            5e6: 0x1A,
            10e6: 0x1C,
            20e6: 0x1E,
            50e6: 0x22,
            100e6: 0x24,
            200e6: 0x28,
            500e6: 0x30,
            800e6: 0x32,
            1000e6: 0x35,
            1200e6: 0x37,
            1500e6: 0x3A,
            1800e6: 0x3D,
        }

        self.input_range_dict = {
            0.040: 0x2,
            0.050: 0x3,
            0.080: 0x4,
            0.100: 0x5,
            0.200: 0x6,
            0.400: 0x7,
            0.500: 0x8,
            0.800: 0x9,
        }

        self.max_pretrigger = 4096

        self.trigger_source_dict = {
            "CHAN_A": 0,
            "CHAN_B": 1,
            "EXTERNAL": 2,
            "DISABLE": 3,
        }

        self.trigger_slope_dict = {"POSITIVE": 1, "NEGATIVE": 2}

        self.clock_source_dict = {
            "INTERNAL": 0x1,
            "EXTERNAL": 0x2,
            "EXTERNAL_CLOCK_10MHz_REF": 0x7,
        }

        self.ext_trig_dict = {"ETR_TTL": 2, "ETR_2V5": 3}

        self.channel_coupling_dict = {"DC": 2}

        # SET DEFAULTS

        self.channel_settings = {"CHAN_A": [0.8, "DC", 50], "CHAN_B": [0.8, "DC", 50]}

        self.trigger_settings = {
            "TRIG_1": ["DISABLE", "POSITIVE", 128],
            "TRIG_2": ["EXTERNAL", "POSITIVE", 128],
        }

        self.ext_trigger_settings = ["DC", "ETR_2V5"]

        self.clock_settings = ["INTERNAL", 1e9, 0]


class ATS9373(atsapi.Board):
    def __init__(self, **kwargs):
        self.name = "ATS9373"

        atsapi.Board.__init__(self, **kwargs)

        self.n_samples_per_chan, self.bits_per_sample = self.getChannelInfo()
        self.n_samples_per_chan = int(self.n_samples_per_chan.value)
        self.bits_per_sample = int(self.bits_per_sample.value)

        self.bytes_per_sample = (self.bits_per_sample + 7) // 8

        self.record_limits = (256, 128)  # minimum length of record, 128 point step size

        self.channel_dict = {"CHAN_A": 1, "CHAN_B": 2}  # Names of the board's channels

        self.channel_impedance_dict = {50: 2}

        self.sample_rate_dict = {
            1e3: 0x1,  # available clock rates
            2e3: 0x2,
            5e3: 0x5,
            10e3: 0x8,
            20e3: 0xA,
            50e3: 0xC,
            100e3: 0xE,
            200e3: 0x10,
            500e3: 0x12,
            1e6: 0x14,
            2e6: 0x18,
            5e6: 0x1A,
            10e6: 0x1C,
            20e6: 0x1E,
            50e6: 0x22,
            100e6: 0x24,
            200e6: 0x28,
            500e6: 0x30,
            800e6: 0x32,
            1000e6: 0x35,
            1200e6: 0x37,
            1500e6: 0x3A,
            1800e6: 0x3D,
            2000e6: 0x3F,
        }

        self.input_range_dict = {
            0.040: 0x2,
            0.050: 0x3,
            0.080: 0x4,
            0.100: 0x5,
            0.200: 0x6,
            0.400: 0x7,
            0.500: 0x8,
            0.800: 0x9,
        }

        self.max_pretrigger = 4096

        self.trigger_source_dict = {
            "CHAN_A": 0,
            "CHAN_B": 1,
            "EXTERNAL": 2,
            "DISABLE": 3,
        }

        self.trigger_slope_dict = {"POSITIVE": 1, "NEGATIVE": 2}

        self.clock_source_dict = {
            "INTERNAL": 0x1,
            "EXTERNAL": 0x2,
            "EXTERNAL_CLOCK_10MHz_REF": 0x7,
        }

        self.ext_trig_dict = {"ETR_TTL": 2, "ETR_2V5": 3}

        self.channel_coupling_dict = {"DC": 2}

        # SET DEFAULTS
        self.channel_settings = {"CHAN_A": [0.8, "DC", 50], "CHAN_B": [0.8, "DC", 50]}

        self.trigger_settings = {
            "TRIG_1": ["DISABLE", "POSITIVE", 128],
            "TRIG_2": ["EXTERNAL", "POSITIVE", 128],
        }

        self.ext_trigger_settings = ["DC", "ETR_2V5"]

        self.clock_settings = [
            "EXTERNAL_CLOCK_10MHz_REF",
            2000e6,
            0,
        ]  # ADDED DECIMATION to this variable


board_dict = {13: ATS9870, 25: ATS9360, 29: ATS9373}


class Alazar:
    """Class for Alazar fast digitzer cards."""

    def __init__(self, name: str = "Alazar"):
        """Initialize the board.

        Server initialization fails if no board can be found.
        """
        self.name = name
        self.buffers = []
        num_boards = atsapi.numOfSystems()
        if num_boards == 0:
            raise ValueError("No Alazar board systems found!")
        # pick the first board from the first system

        board_num = atsapi.Board(systemId=1, boardId=1).type
        self.board = board_dict[board_num](systemId=1, boardId=1)
        print(
            "Alazar model {}, bit depth: {}".format(
                self.board.name, self.board.bits_per_sample
            )
        )

        for chan in self.board.channel_dict.keys():
            self.channel_settings(chan)

        self.trigger_settings()
        self.ext_trigger_settings()
        self.capture_clock_settings()

        # Minimum number of buffers we'll allow. Set this higher to avoid APIBufferOverflow error
        # self.min_buffers = 32 # increased to fix overflow error with ATS9870
        self.min_buffers = 128  # Was 128, JM has 1024

    def reset_boards(self):
        """Tries to forcefully close and reopen the ATS api.

        This should reset the board settings. Probably doesn't work yet.
        """
        # import _ctypes
        # if atsapi.ats is not None:
        #     _ctypes.FreeLibrary(atsapi.ats)

        for chan in self.board.channel_dict.keys():
            self.channel_settings(chan)

        self.trigger_settings()

        self.ext_trigger_settings()

        self.capture_clock_settings()

    def list_boards(self, system=1):
        """List the boards in the given sysyem id, defaults to 1 (usually is only 1)"""
        num_boards = atsapi.boardsInSystemBySystemID(system)
        return f"System {system} has {num_boards} board(s)"

    # settings to return various parameters for this board
    def board_type(self):
        """Return the numeric type of this Alazar board."""
        return self.board.type

    def board_dsp_modules(self):
        """Return the list of dsp modules in the alazar.

        Modules are delineated by ID number, version maj/minor, and length.
        """
        return [x.dspGetInfo() for x in self.board.dspGetModules()]

    def board_model(self):
        """Return the model of this Alazar board."""
        return self.board.name

    def channels(self):
        """Return the channels for this Alazar board."""
        return self.board.channel_dict.keys()

    def channel_memory(self):
        """Return the number of channel for this Alazar board."""
        return int(self.board.n_samples_per_chan)

    def trigger_sources(self):
        """Return the trigger sources for this Alazar board."""
        return self.board.trigger_source_dict.keys()

    def clock_sources(self):
        """Return the clock sources for this Alazar board."""
        return self.board.clock_source_dict.keys()

    def sample_rates(self):
        """Return the valid sample rates for this Alazar board."""
        return sorted(self.board.sample_rate_dict.keys())

    def ranges(self):
        """Return the valid ranges for this Alazar board."""
        return sorted(self.board.input_range_dict.keys())

    def input_couplings(self):
        """Return the valid input couplings for this Alazar board."""
        return self.board.channel_coupling_dict.keys()

    def ext_trig_range(self):
        """Return the valid external trigger ranges for this Alazar board."""
        return self.board.ext_trig_dict.keys()

    def channel_settings(
        self,
        channel: str,
        input_range: float = 0.4,
        coupling: str | None = None,
        impedance: str | None = None,
    ):
        """Sets channel input range, coupling and impedance.

        Args:
        channel: which channel to set set. See your Alazar's channel_dict for allowed values
        input range: input signal range (selects an onboard amplifier). See input_range_dict
        channel coupling: AC, DC etc. See channel_coupling_dict for allowed values
        impedance: see channel_impedance_dict for allowed values

        Raises AlazarException for invalid parameters
        """
        if channel not in self.board.channel_dict.keys():
            raise Exception(
                "No Channel of name {}, available channels are {}".format(
                    channel, self.board.channel_dict.keys()
                )
            )

        if input_range is not None:
            if input_range in self.board.input_range_dict.keys():
                self.board.channel_settings[channel][0] = input_range
            else:
                raise Exception(
                    "Board not capable of {} range, can only do {}".format(
                        input_range, self.board.input_range_dict.keys()
                    )
                )

        if coupling is not None:
            if coupling in self.board.channel_coupling_dict.keys():
                self.board.channel_settings[channel][1] = coupling
            else:
                raise Exception(
                    "Board not capable of {} coupling, can only do {}".format(
                        coupling, self.board.channel_coupling_dict.keys()
                    )
                )

        if impedance is not None:
            if impedance in self.board.channel_impedance_dict.keys():
                self.board.channel_settings[channel][2] = impedance
            else:
                raise Exception(
                    "Board not capable of {} impedance, can only do {}".format(
                        impedance, self.board.channel_impedance_dict.keys()
                    )
                )

        self.board.inputControl(
            self.board.channel_dict[channel],
            self.board.channel_coupling_dict[self.board.channel_settings[channel][1]],
            self.board.input_range_dict[self.board.channel_settings[channel][0]],
            self.board.channel_impedance_dict[self.board.channel_settings[channel][2]],
        )

        return self.board.channel_settings[channel]

    def trigger_settings(
        self,
        trig_source: str | None = None,
        trig_slope: str | None = None,
        trig_threshold: int | None = None,
    ):
        """Sets the source of the trigger to start recording a Record

        Args:
            trig_source: sets which alazar input should be used to trigger. Allowed
                values given by trigger_source_dict
            trig_slope: trigger on rising or falling edge. Allowed values given by
                trigger_slope_dict
            trig_threshold: threshold for triggering, from 0 to 255. 0V is 128.

        Returns:
            Trigger_settings info array
        """
        trigger = "TRIG_2"  # TRIG 1 doesnt seem to do anything, need to consult the documentation more
        # for now we just eliminate the functionality of it

        if trigger not in self.board.trigger_settings.keys():
            raise Exception(
                "No trigger named {}, available triggers are {}".format(
                    trigger, self.board.trigger_settings.keys()
                )
            )

        if trig_source is not None:
            if trig_source in self.board.trigger_source_dict.keys():
                self.board.trigger_settings[trigger][0] = trig_source
            else:
                raise Exception(
                    "Board does not support {} as a trigger source, allowed sources are {}".format(
                        trig_source, self.board.trigger_source_dict.keys()
                    )
                )

        if trig_slope is not None:
            if trig_slope in self.board.trigger_slope_dict.keys():
                self.board.trigger_settings[trigger][1] = trig_slope
            else:
                raise Exception(
                    "Board does not support {} as a trigger slope, allowed slopes are {}".format(
                        trig_slope, self.board.trigger_slope_dict.keys()
                    )
                )

        if trig_threshold is not None:
            if trig_threshold > 0 and trig_threshold < 255:
                self.board.trigger_settings[trigger][2] = int(trig_threshold)
            else:
                raise Exception(
                    "Trigger threshold must be between 0 and 255, where 128 is 0V"
                )

        self.board.setTriggerOperation(
            0,
            0,
            self.board.trigger_source_dict[self.board.trigger_settings["TRIG_1"][0]],
            self.board.trigger_slope_dict[self.board.trigger_settings["TRIG_1"][1]],
            self.board.trigger_settings["TRIG_1"][2],
            0,
            self.board.trigger_source_dict[self.board.trigger_settings["TRIG_2"][0]],
            self.board.trigger_slope_dict[self.board.trigger_settings["TRIG_2"][1]],
            self.board.trigger_settings["TRIG_2"][2],
        )

        return self.board.trigger_settings[trigger]

    def ext_trigger_settings(self, coupling=None, ext_range=None):
        if coupling is not None:
            if coupling in self.board.channel_coupling_dict.keys():
                self.board.ext_trigger_settings[0] = coupling
            else:
                raise Exception(
                    "External Trigger does not accept {} coupling, must be {}".format(
                        coupling, self.board.channel_coupling_dict.keys()
                    )
                )

        if ext_range is not None:
            if ext_range in self.board.ext_trig_dict.keys():
                self.board.ext_trigger_settings[1] = ext_range
            else:
                raise Exception(
                    "External Trigger does not accept {} as a range, must be {}".format(
                        ext_range, self.board.ext_trig_dict.keys()
                    )
                )

        self.board.setExternalTrigger(
            self.board.channel_coupling_dict[self.board.ext_trigger_settings[0]],
            self.board.ext_trig_dict[self.board.ext_trigger_settings[1]],
        )

        return self.board.ext_trigger_settings

    # settings to set up acquisition parameters, except decimation
    def capture_clock_settings(
        self, clock_source: str | None = None, sample_rate: str | None = None
    ):
        """Set the capture clock for this alazar board.

        Args:
            clock_source: the name of a valid clock source for this board
            sample_rate: the name of a valid sample rate for this board, or the
                numeric sample rate in MHz for the case of 9360 10 MHz PLL clock
                For the 9870, if clock_source is not "internal", sample_rate is ignored

        Raises:
            AlazarException for invalid parameters, or if the set clock call fails.
        """

        if clock_source is not None:
            if clock_source in self.board.clock_source_dict.keys():
                self.board.clock_settings[0] = clock_source
            else:
                raise Exception(
                    "Board cannot use clock {}, must be {}".format(
                        clock_source, self.board.clock_source_dict.keys()
                    )
                )

        if sample_rate is not None:
            if (
                sample_rate in self.board.sample_rate_dict.keys()
                or self.check_clock_source(clock_source)
            ):
                self.board.clock_settings[1] = sample_rate
            else:
                raise Exception(
                    "Board does not support sample rate {}, must be one of {}".format(
                        sample_rate, sorted(self.board.sample_rate_dict.keys())
                    )
                )

        # Handle sample rate differently depending on if we are generating the clock internally or externally
        if self.check_clock_source(self.board.clock_settings[0]):
            self.board.setCaptureClock(
                self.board.clock_source_dict[self.board.clock_settings[0]],
                self.board.clock_settings[
                    1
                ],  # Check ATS-SDK user manual for allowed values
                atsapi.CLOCK_EDGE_RISING,
                self.board.clock_settings[2],
            )
        else:
            self.board.setCaptureClock(
                self.board.clock_source_dict[self.board.clock_settings[0]],
                self.board.sample_rate_dict[self.board.clock_settings[1]],
                atsapi.CLOCK_EDGE_RISING,
                self.board.clock_settings[2],
            )

        return self.board.clock_settings

    def check_clock_source(self, clock_source):
        """Checks the clock source.

        Based on the clock_source, check to see if the ATS API is expecting a hex code or
        a frequency in Hz. If this function throws an error, it's likely that you tried
        to set the clock source to something that has not been added to
        `board.clock_source_dict` or is not availble for the board you are using.

        Returns:
            `0` if sample rate must be restricted to sample_rate_dict values or `1` if
            sample rate should be provided as frequency in Hz.
        """
        if (
            self.board.clock_source_dict[clock_source]
            == atsapi.EXTERNAL_CLOCK_10MHz_REF
        ):
            return 1
        return 0

    def capture_decimation(self, decimation: int | None = None):
        """Set decimation.

        Zero means no decimation. Note that decimation simply throws away data, rather
        than averaging.

        Args:
            decimation: Take only every nth data point
        """
        if decimation is not None:
            self.board.clock_settings[2] = decimation
        # Handle sample rate differently depending on if we are generating the clock internally or externally
        if self.check_clock_source(self.board.clock_settings[0]):
            self.board.setCaptureClock(
                self.board.clock_source_dict[self.board.clock_settings[0]],
                self.board.clock_settings[
                    1
                ],  # Check ATS-SDK user manual for allowed values
                atsapi.CLOCK_EDGE_RISING,
                self.board.clock_settings[2],
            )
        else:
            self.board.setCaptureClock(
                self.board.clock_source_dict[self.board.clock_settings[0]],
                self.board.sample_rate_dict[self.board.clock_settings[1]],
                atsapi.CLOCK_EDGE_RISING,
                self.board.clock_settings[2],
            )

        return self.board.clock_settings

    def get_true_sample_rate(self):
        """Get the sampling rate of the board.

        This takes into account both the clock rate and the decimation setting.

        Returns:
            sampling rate in Hz
        """
        current_clock_settings = self.capture_decimation()
        if current_clock_settings[2]:  # if decimation is occuring
            return current_clock_settings[1] / current_clock_settings[2]
        else:
            return current_clock_settings[1]

    def start_acquire(
        self,
        samples: int,
        n_repetitions: int = 1,
        mean: bool = False,
        pre_trigger: int = 0,
        expected_triggers: int = 1,
        mode: str = "TRIG",
        save_file_path: str | None = None,
        save_only: bool = False,
    ):
        """Starts an acquisition.

        Prepares the alazar to receive data. Will respond to
        `n_repetitions*expected_triggers` triggers and save as many records, each
        containing samples time points. Assumes data are acquired using an AWG which
        cycles through expected_triggers distinct sequences. Set expected_triggers=1
        to ignore this functionality.

        Args:
            samples: Number of time points per time trace (record). Minimal value and
                increment given in record_limits.
            n_repetitions: Number of times to record a time trace for each AWG sequence
                element.
            mean: Average traces. Does so in a memory-efficient way.
            pre_trigger: Include this many time points before the trigger ocurred.
                Limits are board-specific. (Note: currently passing ADMA_NPT=ADMA
                NoPreTrigger, so pretrigger may not be working).
            expected_triggers: Number of sequence elements.
            mode: 'TRIG' only acquires upon trigger. Otherwise acquires continuously.
            save_file_path: If not None, streams Alazar data directly to harddrive.
                Currently ignored if mode is not `'TRIG'`.
            save_only: If save_file_path is not None, you can get faster streaming to
                hard drive by setting this to True, which which case no data is passed
                to python.
        """
        n_channels = len(self.board.channel_dict)
        self.expected_triggers = expected_triggers
        self.n_repetitions = n_repetitions
        self.mean = mean
        self.save_file_path = save_file_path
        self.save_only = save_only
        self.samples = max(samples, self.board.record_limits[0])

        if self.samples % self.board.record_limits[1] != 0:
            self.samples = (
                self.samples
                - self.samples % self.board.record_limits[1]
                + self.board.record_limits[1]
            )

        if self.board.bytes_per_sample == 2:
            buffer_type = ctypes.c_uint16
            data_type = "uint16"
        elif self.board.bytes_per_sample == 1:
            buffer_type = ctypes.c_uint8
            data_type = "uint8"
        else:
            print(self.board.bytes_per_sample)
            raise Exception("Board has more than 2 bytes per sample??")

        if pre_trigger is not None:
            if pre_trigger >= 0 and pre_trigger < self.board.max_pretrigger:
                if pre_trigger % self.board.record_limits[1] != 0:
                    pre_trigger = (
                        pre_trigger
                        - pre_trigger % self.board.record_limits[1]
                        + self.board.record_limits[1]
                    )
            else:
                raise Exception(
                    "Pre-Trigger length out of range, must be between 0 and {}".format(
                        self.board.max_pretrigger
                    )
                )

        self.board.setRecordSize(pre_trigger, samples - pre_trigger)
        self.board.setTriggerTimeOut(0)

        if mode == "TRIG":
            if self.save_file_path is not None:
                AutoCreateBuffersFlag = (
                    atsapi.ADMA_ALLOC_BUFFERS
                )  # Let the atsapi automatically post buffers. Necessary for streaming to hard drive
            else:
                AutoCreateBuffersFlag = 0  # (bitwise or with 0 has no effect)
            self.board.beforeAsyncRead(
                2**n_channels
                - 1,  # This is binary 11111 where the number of 1s is the number of channels
                -pre_trigger,  # pre-trigger samples
                samples,  # samples per record
                self.expected_triggers,  # records per buffer
                (self.n_repetitions + 1) * self.expected_triggers,
                # records per acquisition. This argument is ignored unless ADMA_ALLOC_BUFFERS flag is on
                # For some reason, this has to be n_repetitions+1 to work. No idea why.
                atsapi.ADMA_EXTERNAL_STARTCAPTURE
                | atsapi.ADMA_NPT
                | atsapi.ADMA_FIFO_ONLY_STREAMING
                | AutoCreateBuffersFlag,
            )  # flag
            if self.save_file_path is not None:
                self.board.createStreamFile(self.save_file_path)
        else:
            self.board.beforeAsyncRead(
                2**n_channels
                - 1,  # This is binary 11111 where the number of 1s is the number of channels
                -pre_trigger,  # pre-trigger samples
                samples,  # samples per record
                self.expected_triggers,  # records per buffer
                self.n_repetitions * self.expected_triggers,  # records per acquisition
                atsapi.ADMA_EXTERNAL_STARTCAPTURE
                | atsapi.ADMA_CONTINUOUS_MODE
                | atsapi.ADMA_FIFO_ONLY_STREAMING,
            )  # flag

        if self.save_file_path is not None and self.save_only:
            self.bytes_to_copy = 0
            bytes_per_buffer = 1  # allocate minimal buffers. They won't be used.
            self.data = np.ndarray((0, 0, 0, 0))  # empty array
        else:
            bytes_per_buffer = (
                samples
                * n_channels
                * self.board.bytes_per_sample
                * self.expected_triggers
            )  # *self.buffer_step_size
            self.bytes_to_copy = bytes_per_buffer
            if not self.mean:
                self.data = np.ndarray(
                    (n_channels, n_repetitions, self.expected_triggers, samples),
                    dtype=data_type,
                )
            else:
                self.data = np.zeros(
                    (n_channels, 1, self.expected_triggers, samples), dtype="float64"
                )

        self.buffers = []

        for i in range(min(self.n_repetitions, self.min_buffers)):
            self.buffers.append(atsapi.DMABuffer(buffer_type, bytes_per_buffer))

        # If streaming to hard drive is turned on, Alazar API automatically posts buffers (ADMA_ALLOC_BUFFERS handle)
        if self.save_file_path is None:
            for buffer in self.buffers:
                self.board.postAsyncBuffer(buffer.addr, buffer.size_bytes)

        try:
            while self.board.busy():  # wait until the board is not busy
                time.sleep(0.01)

            self.board.startCapture()  # Start the acquisition
        except:
            self.stop()

    def acquire(self, timeout: int = 10000, convert_to_float: bool = False):
        """Acquire data from the alazar. Returns a 4 dimensional array of data

        Args:
            timeout: time to wait for trigger before cancelling (ms)
            convert_to_float: returns data as a numpy float if `True`, otherwise returns
                an integer from zero to 2^bit depth.

        Returns:
            An array of size `channels x n_repetitions x expected_triggers x samples`
            if `mean=False`, or `channels x 1 x expected_triggers x samples` if
            `mean=True`.
        """

        current_buffer = 0
        while current_buffer < self.n_repetitions:
            try:
                for buffer in self.buffers:
                    # Wait for the buffer at the head of the list of available
                    # buffers to be filled by the board.
                    if self.save_file_path is None:
                        self.board.waitAsyncBufferComplete(
                            buffer.addr, timeout_ms=int(timeout)
                        )
                    else:
                        self.board.waitNextAsyncBufferComplete(
                            buffer.addr, self.bytes_to_copy, timeout_ms=int(timeout)
                        )

                    if self.save_only:
                        pass
                    elif not self.mean:
                        self.data[0][
                            current_buffer : (1 + current_buffer)
                        ] = np.reshape(
                            buffer.buffer[0::2],
                            (1, self.expected_triggers, self.samples),
                        )
                        self.data[1][
                            current_buffer : (1 + current_buffer)
                        ] = np.reshape(
                            buffer.buffer[1::2],
                            (1, self.expected_triggers, self.samples),
                        )
                    else:
                        np.add(
                            np.float64(
                                np.reshape(
                                    buffer.buffer[0::2],
                                    (self.expected_triggers, self.samples),
                                )
                            ),
                            self.data[0][0],
                            self.data[0][0],
                        )
                        np.add(
                            np.float64(
                                np.reshape(
                                    buffer.buffer[1::2],
                                    (self.expected_triggers, self.samples),
                                )
                            ),
                            self.data[1][0],
                            self.data[1][0],
                        )

                    # Add the buffer to the end of the list of available buffers.
                    if self.save_file_path is None:
                        self.board.postAsyncBuffer(buffer.addr, buffer.size_bytes)
                    current_buffer += 1
                    if current_buffer >= self.n_repetitions:
                        break
            except:
                self.stop()
                raise
        self.stop()

        if not self.save_only:
            if self.mean:
                self.data = self.data / self.n_repetitions

            if convert_to_float:
                self.data = (
                    np.float64(self.data) / 2.0 ** (self.board.bytes_per_sample * 8 - 1)
                    - 1
                )

        return self.data

    def stop(self):
        for buffer in self.buffers:
            buffer.__exit__()
        self.buffers = []
        self.board.abortCapture()
        self.board.abortAsyncRead()
