# Copyright (c) 2024-25, UC Regents

from qwip.instruments.zurich.discovery import ZIDiscovery
from qwip.instruments.zurich.device_setup import create_setup
from qwip.instruments.zurich.devices import (
    HDAWG8,
    SHFQC,
    UHFQA,
)

# convenience short-cuts
HDWAG = HDAWG8
