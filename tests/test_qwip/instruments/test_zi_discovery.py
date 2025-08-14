import pytest

from qwip.instruments.zurich import (
    ZIDiscovery,
)


class TestDiscovery:
     def teardown_method(self, method):
       # destroy the singleton to guarantee a fresh session in each method;
       # there's no API as this shouldn't be done in normal running; it's
       # fine here as the tests run offline
         ZIDiscovery._discovery = None

     def test_discovery_session(self):
         """Create a default ZI discovery session."""

         disc1 = ZIDiscovery()
         disc2 = ZIDiscovery()

         assert disc1 == disc2    # singleton

         disc1.close()

     def test_simulated_discovery(self):
         """Create a default simulated ZI discovery."""

         for exptype in ["", # default of SHFQC
                         "MultiQubitBasic",
                         "BlizzardPQSC",
                         "SHFQCSingle"]:
             disc = ZIDiscovery(exptype=exptype, simulation=True)

             devices = disc.get_devices()
             assert devices

             if "SHFQC" in exptype:
                 assert "SHFQC" in devices
             elif exptype:
                 if "PQSC" in exptype:
                     assert "PQSC" in devices
                 assert "HDAWG" in devices
                 assert "UHFQA" in devices

             setup = disc.get_setup()
             assert setup

             disc.close()
             ZIDiscovery._discovery = None

