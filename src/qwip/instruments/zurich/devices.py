# Copyright (c) 2024, UC Regents

"""LabOneQ device representations with reasonable defaults.
"""

import io
from .device_options import (
    DevicePurpose,
    get_device_options,
    get_device_nchannels,
)
from laboneq.simple import (
    DeviceSetup,
)

__all__ = [
    "HDAWG8",
    "HDAWG",
    "PQSC",
    "SHFQC",
    "UHFQA",
]


generic_instruments_descr = """\
  {kind_u}:
    - address: {name}
      uid: device_{kind_l}_{name}
      options: {kind_o}
"""

# ports descriptors vary by device and (typical) use
ports_iq_drive_descr = {
    "hdawg8": """\
    - iq_signal: q{qubit}/drive
      ports: [SIGOUTS/{i}, SIGOUTS/{q}]
""",
    "shfqc": """\
    - iq_signal: q{qubit}/drive
      ports: SGCHANNELS/{qubit}/OUTPUT
""",
}
ports_iq_drive_descr["hdawg"] = ports_iq_drive_descr["hdawg8"]
ports_iq_drive_descr["uhfqa"] = ports_iq_drive_descr["hdawg8"]

ports_single_drive_descr = {
    "hdawg8": """\
    - rf_signal: q{qubit}/drive
      ports: SIGOUTS/{i}
""",
    "shfqc": """\
    - rf_signal: q{qubit}/drive
      ports: SGCHANNELS/{qubit}/OUTPUT
""",
}
ports_single_drive_descr["hdawg"] = ports_single_drive_descr["hdawg8"]

ports_measure = {
    "shfqc": """\
    - iq_signal: q{qubit}/measure
      ports: QACHANNELS/0/OUTPUT
""",
    "uhfqa": """\
    - iq_signal: q{qubit}/measure
      ports: [SIGOUTS/0, SIGOUTS/1]
"""
}

ports_acquire = {
    "shfqc": """\
    - acquire_signal: q{qubit}/acquire
      ports: QACHANNELS/0/INPUT
""",
    "uhfqa": """\
    - acquire_signal: q{qubit}/acquire
""",
}


class _DescriptorMixin:
    """Create descriptor sections for a ZI device"""

    def __init__(self, kind: str, name: str, order: int):
        self.kind  = kind
        self.name  = name
        self.order = order

    def instrument_descr(self):
        labels = {"kind_u": self.kind.upper(),
                  "kind_l": self.kind.lower(),
                  "kind_o": get_device_options(self.kind, self.name),
                  "name":   self.name}
        return generic_instruments_descr.format(**labels)

    def _channel_descr(self, template):
        nchannels, iq_paired = get_device_nchannels(self.kind, DevicePurpose.DRIVE)
        if not nchannels:
            return ""

        grouping = 1
        if not iq_paired:
            grouping = 2
            nchannels = nchannels // 2

        descriptor = io.StringIO()
        for iq in range(nchannels):
            descriptor.write(template.format(qubit=iq, i=iq*grouping, q=iq*grouping+1))

        descriptor.seek(0)
        return descriptor.read()

    def drive_descr(self, grouping: str = "iq"):
        _kind = self.kind.lower()

        if grouping == "iq":
            template = ports_iq_drive_descr.get(_kind, "")
        elif grouping == "single":
            template = ports_single_drive_descr.get(_kind, "")
        else:
            raise RuntimeError(f"unknown grouping \"{grouping}\" (acceptable values: iq, single)")

        return self._channel_descr(template)

    def measure_descr(self, grouping: str = "iq"):
        template = ports_measure.get(self.kind.lower(), "")
        return self._channel_descr(template)

    def acquire_descr(self, grouping: str = "iq"):
        template = ports_acquire.get(self.kind.lower(), "")
        return self._channel_descr(template)


class HDAWG8(_DescriptorMixin):
    def __init__(self, name: str, order: int = 0):
        super(HDAWG, self).__init__("hdawg", name, order)

HDAWG = HDAWG8


class UHFQA(_DescriptorMixin):
    def __init__(self, name, order=0):
        super(UHFQA, self).__init__("uhfqa", name, order)


shfc_connections_descr = """\
connections:
  device_shfqc:

    - iq_signal: {q0}/measure
      ports: [QACHANNELS/0/OUTPUT]
    - acquire_signal: {q0}/acquire
      ports: [QACHANNELS/0/INPUT]
    - iq_signal: {q1}/measure
      ports: [QACHANNELS/0/OUTPUT]
    - acquire_signal: {q1}/acquire
      ports: [QACHANNELS/0/INPUT]
    - iq_signal: {q2}/measure
      ports: [QACHANNELS/0/OUTPUT]
    - acquire_signal: {q2}/acquire
      ports: [QACHANNELS/0/INPUT]
    - iq_signal: {q3}/measure
      ports: [QACHANNELS/0/OUTPUT]
    - acquire_signal: {q3}/acquire
      ports: [QACHANNELS/0/INPUT]
    - iq_signal: {q4}/measure
      ports: [QACHANNELS/0/OUTPUT]
    - acquire_signal: {q4}/acquire
      ports: [QACHANNELS/0/INPUT]
    - iq_signal: {q5}/measure
      ports: [QACHANNELS/0/OUTPUT]
    - acquire_signal: {q5}/acquire
      ports: [QACHANNELS/0/INPUT]
"""

class SHFQC(_DescriptorMixin):
    def __init__(self, name, order=0):
        super(SHFQC, self).__init__("shfqc", name, order)


class PQSC(_DescriptorMixin):
    def __init__(self, name, order=0):
        super(PQSC, self).__init__("PQSC", name, order)

