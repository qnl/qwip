# Copyright (c) 2024, UC Regents

"""Registry of options of ZI devices.
"""

from enum import Enum


__all__ = [
    "get_device_options",
    "get_device_nchannels",
]

_known_devices = {

}


class DevicePurpose(Enum):
    DRIVE   = 0
    MEASURE = 1    # ZI terminology
    READOUT = 1    # id. QWiP
    ACQUIRE = 2    # ZI
    DEMOD   = 2    # QWiP

DEVICE_TYPES = {
    DevicePurpose.DRIVE:   ["hdawg", "shfqc", "shfsg"],
    DevicePurpose.MEASURE: ["shfqc", "shfqa", "uhfqa"],
    DevicePurpose.ACQUIRE: ["shfqc", "shfqa", "uhfqa"],
}

DEVICE_SIGNAL_TYPES = {
    DevicePurpose.DRIVE:   "drive",
    DevicePurpose.MEASURE: "measure",
    DevicePurpose.ACQUIRE: "acquire",
}

# number of channels by device type; tuple represents (drive, measure, acquire)
_device_channels = {
    "hdawg8": (8, 0,  0),
    "pqsc"  : (0, 0,  0),
    "shfqc" : (6, 2,  6),
    "uhfqa" : (0, 2, 10),
}
_device_channels["hdawg"] = _device_channels["hdawg8"]


def get_device_options(kind: str, name: str = None) -> str:
    """Retrieve installed device options.

    Return the list of installed options for known devices.

    Args:
        kind: device type (e.g. "SHFQC")
        name: device address (if known; e.g. "dev8159")

    Returns:
        LabOneQ-formatted string listing device options
    """

    options = _known_devices.get(name, None)
    if options is not None:
        return options

  # either emulated or not registered yet
    _kind = kind.lower()
    if _kind == "hdawg":
        return "HDAWG8/MF/ME/SKW/PC"
    elif _kind == "shfqc":
        return "SHFQC/QC6CH"

    return kind


def get_device_nchannels(kind: str, purpose: DevicePurpose) -> (int, bool):
    """Get the number of channels for a given device type and purpose.

    Args:
        kind: device type (e.g. "SHFQC")
        purpose: device channel usage (e.g. DevicePurpose.DRIVE)

    Returns:
        Integer with the max number of available channels
        Boolean indicating whether IQ channels are paired (true) or separate (false)
    """

    _kind = kind.lower()
    return _device_channels[_kind][purpose.value], _kind not in ["hdawg", "uhfqa"]

