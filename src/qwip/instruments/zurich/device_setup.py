# Copyright (c) 2025, UC Regents

"""Zurich Instruments' device setup.

Prebuilt experiments (in the LabOneQ definition of the word), consisting of setups
of connected devices, using logical defaults.
"""

import io

from typing import Dict
from laboneq.dsl.device.device_setup import DeviceSetup
from .devices import (
    HDAWG,
    HDAWG8,
    PQSC,
    SHFQC,
    UHFQA,
)

__all__ = [
    "create_setup"
]


MultiQubitBasic = {"hdawg8", "uhfqa"}
SHFQCSingle     = {"shfqc"}
BlizzardPQSC    = {"hdawg8", "uhfqa", "pqsc"}

_canonical_setup_name = {
    "shfqcsingle"    : "SHFQCSingle",
    "multiqubitbasic": "MultiQubitBasic",
    "blizzardpqsc"   : "BlizzardPQSC",
}


def _open_device_section(isopen, descriptor, device):
    if not isopen:
        descriptor.write("  device_%s_%s:\n" % (device.kind.lower(), device.name))
    return True

def _create_device_setup(devices_by_kind: Dict[str, list]):
    devices = list()
    for kind, names in devices_by_kind.items():
        for idx, name in enumerate(names):
            devices.append(eval(kind.upper())(name, order=idx))

    assert devices and "a setup requires at least one device"

    descriptor = io.StringIO()
    descriptor.write("instruments:\n")
    for device in devices:
        descriptor.write(device.instrument_descr())

    descriptor.write("\nconnections:\n")
    for device in devices:
        section_opened = False

        for get_descr in ['drive_descr', 'measure_descr', 'acquire_descr']:
            descr = getattr(device, get_descr)(grouping="iq")
            if descr:
                section_opened = _open_device_section(section_opened, descriptor, device)
                descriptor.write(descr)

    descriptor.seek(0)
    return descriptor.read()


def create_setup(devices: Dict[str, list],
                 exptype: str|None = None,
                 server_host: str = "localhost",
                 server_port: int = 8004):
    """Create an experimental setup from the given devices.

    Setup the given devices either by prebuilt experimental setup or, if None,
    by the most common setup given the provided devices.

    Args:
        devices: map of device ids by kind
        exptype: experimental wiring to use or regular if None

    Returns:
        LabOneQ Experiment
    """

    _devices = {key.lower(): devices[key] for key in devices.keys()}
    devtypes = set(_devices.keys())
    try:
        devtypes.remove("hdawg")
        devtypes.add("hdawg8")
    except KeyError:
        pass

    if exptype is None:
      # no explicit experiment type given; pick the most common setup based
      # on the provided devices

        if not devtypes.difference(MultiQubitBasic):
            exptype = "multiqubitbasic"
        elif not devtypes.difference(SHFQCSingle):
            shfqcs = _devices["shfqc"]
            if isinstance(shfqcs, str) or len(shfqcs) == 1:
                exptype = "shfqcsingle"
        elif not devtypes.difference(BlizzardPQSC):
            exptype = "blizzardpqsc"
        else:
            raise RuntimeError("could not derive experiment type from the given devices")
    else:
        exptype = exptype.lower()

        if exptype not in _canonical_setup_name.keys():
            raise RuntimeError(f"unknown experiment type \"{exptype}\"")

        if devtypes.difference(eval(_canonical_setup_name[exptype])):
            raise RuntimeError(f"incorrect device types for experiment type \"{exptype}\"")

  # construct the requested experiment setup
    device_setup = DeviceSetup.from_descriptor(
        _create_device_setup(_devices),
        server_host=server_host,
        server_port=server_port,
        setup_name=_canonical_setup_name[exptype]
    )

    return device_setup

