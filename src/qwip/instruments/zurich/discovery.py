# Copyright (c) 2024-25, UC Regents

"""Zurich Instruments' self-discovery.

If a data server is only associated with a single experimental setup (this
is common if the data server runs on localhost), they can be "discovered"
automatically as opposed to explicitly defining device addresses.
"""

import logging
import zhinst.core

from typing import Dict, List
from collections import defaultdict

from .device_setup import create_setup


logger = logging.getLogger(__name__)

logging2daqlogging = {
    logging.FATAL   : 5,
    logging.ERROR   : 4,
    logging.WARNING : 3,
    logging.INFO    : 1,
    logging.DEBUG   : 2 } # trace:0, status:6


class ZIDiscovery:
    """Internal representation of a ZI device discovery; a singleton to allow
    all discovery operations to stay within a single session, saving time.
    """

    _discovery = None

    def __new__(cls, *args, **kwds):
        if cls._discovery is None:
            if kwds.get('simulation', False):    # for testing purposes
                zi = object.__new__(_SimulatedZIDiscovery)
                _SimulatedZIDiscovery._singleton__init__(zi, *args, **kwds)
            else:
                zi = object.__new__(cls)
                cls._singleton__init__(zi, *args, **kwds)
            cls._discovery = zi
        return cls._discovery

    def __init__(self, *args, **kwds):
        pass

    def _singleton__init__(self, *args, **kwds):
        self._zi_discovery  = None     # zicore discovery
        self._daq_server    = None     # zicore DAQ server

    def _get_discovery(self):
        if self._zi_discovery is None:
            self._zi_discovery = zhinst.core.ziDiscovery()
        return self._zi_discovery

    def _get_daq(self, devices: List[str] = [], req_apilevel: int = 6):
        """Open a session on the DataServer and return the connection

        Returns:
            daq: zhinst DAQ connection to the data server
        """

        if self._daq_server is None:

          # assume properties are the same for all devices (type already checked)
            discovery = self._get_discovery()

            try:
                daq_props = discovery.get(devices[0])
            except (IndexError, RuntimeError):
              # if no devices, the session will likely fail further downstream anyway, so an
              # attempt to proceed with the defaults here is probably helpful for debugging
                daq_props = {'serveraddress' : '127.0.0.1', 'serverport' : 8004, 'interfaces' : ['1GbE'] }

            logger.info('connecting at %s:%s', daq_props['serveraddress'], daq_props['serverport'])
            self._daq_server = zhinst.core.ziDAQServer(
                daq_props['serveraddress'], daq_props['serverport'], req_apilevel)

          # propagate python logging level to daq logging
            self._daq_server.setDebugLevel(logging2daqlogging[logger.getEffectiveLevel()])

            return self._daq_server

        return self._daq_server

    def get_devices(self, devices: Dict[str, List[str]] = {}) -> Dict[str, List[str]]:
        """Get connected devices.

        Retrieve all named ZI devices of the requested type, or all such devices
        if no names were given.

        Args:
            devices: mapping of device {type, [addresses]} to select

        Returns:
             A list of device ids matching the requested type and names
        """

        discovery = self._get_discovery()
        all_devices = set([x.lower() for x in discovery.findAll()])

      # subset the available devices if explicit names were given
        if devices:
            for devtype, names in devices.items():
                required = set([x.lower() for x in names])
                available = required.intersection(all_devices)
                if available != required:
                    for devid in required:
                        if not devid in all_devices:
                            logger.error("missing device: %s", devid)
                    raise RuntimeError("not all requested devices are available")

              # we're only interested in 'devtype' devices
                for devid in available:
                    props = discovery.get(devid)
                    if props['devicetype'].lower() != devtype.lower():
                        logger.error("device: %s is of incorrect type %s", devid, props['devicetype'])
                        raise RuntimeError("incorrect device type")

                logger.info('selected %s devices: %s', devtype, str(available))

                return devices

      # otherwise collect all available devices by type
        else:
            devices = defaultdict(list)
            for devid in all_devices:
                props = discovery.get(devid)
                devices[props['devicetype'].upper()].apppend(devid)

            for devtype, names in devices.items():
                logger.info('selected %s devices: %s', devtype, str(names))

            return dict(devices)

    def get_setup(self):
        """Get a LabOneQ-compatible setup for the currently connected devices.
        """
        return create_setup(devices=self.get_devices())

    def close(self, devices: List[str] = []):
        """End session with the data server for the given devices. This will
        completely remove the connection, also for any open LabOne sessions on
        the same IP address.

        Args:
            devices: list of device addresses to disconnect (e.g. ["dev8159"])

        Returns:
            None
        """

        if self._daq_server is None:   # no session was ever connected
            return

        daq = self._get_daq()
        for device_id in devices:
            daq.disconnectDevice(device_id)

        daq.disconnect()
        self._daq_server = None        # forces a reset if re-opened


class _SimulatedDAQ(object):
    """Fake ZI DAQ class to allow tests to proceed offline.
    """

    def disconnectDevice(self, device_id):
        pass

    def disconnect(self):
        pass


class _SimulatedZIDiscovery(ZIDiscovery):
    """Fake ZIDiscovery class to allow tests to proceed offline.
    """

    def __init__(self, *args, **kwds):
        pass

    def _singleton__init__(self, *args, **kwds):
        self._daq_server = None
        self._exptype = kwds.get("exptype", "")

    def _get_daq(self):
        if self._daq_server is None:
            self._daq_server = _SimulatedDAQ()
        return self._daq_server

    def get_devices(self, devices: Dict[str, List[str]] = {}) -> Dict[str, List[str]]:
        if devices:
            return devices

        exptype = self._exptype.lower()
        if exptype == "multiqubitbasic":
            return {'HDAWG'  : ['dev0001', 'dev0002'],
                    'UHFQA'  : ['dev1002']}
        elif exptype == "blizzardpqsc":
            return {'HDAWG'  : ['dev0001', 'dev0002'],
                    'UHFQA'  : ['dev1002'],
                    'PQSC'   : ['dev2001']}
        elif exptype == "shfqcsingle":
            return {'SHFQC'  : ['dev2001']}

        return {'SHFQC'  : ['dev2001']}

    def compile(self, exp, settings=None):
        pass

    def run(self, exp):
        pass

