import pytest

"""
_canonical_setup_name = {
    "shfqcsingle"    : "SHFQCSingle",
        "multiqubitbasic": "MultiQubitBasic",
            "blizzardpqsc"   : "BlizzardPQSC",
            }

def create_setup(devices: Dict[str, list],
                 exptype: str|None = None,
                                  server_host: str = "localhost",
                                                   server_port: int = 8004):

"""

from qwip.instruments.zurich import (
    create_setup,
)


class TestSetup:
     def test_known_configurations(self):
         """Test creation of QTRoL ZurichTrigger configurations."""

         mb_devices = {"hdawg": "dev4444", "uhfqa": "dev6666"}
         assert create_setup(devices=mb_devices)
         assert create_setup(devices=mb_devices, exptype="MultiQubitBasic")

         mb_devices = {"hdawg": ["dev4444", "dev5555"], "uhfqa": ["dev6666"]}
         assert create_setup(devices=mb_devices)
         assert create_setup(devices=mb_devices, exptype="MultiQubitBasic")

         s_devices = {"shfqc": "dev1234"}
         assert create_setup(devices=s_devices)
         assert create_setup(devices=s_devices, exptype="SHFQCSingle")

         bp_devices = {"pqsc": "dev1001", "hdawg": "dev5555", "uhfqa": "dev6666"}
         create_setup(devices=bp_devices)
         create_setup(devices=bp_devices, exptype="BlizzardPQSC")

         with pytest.raises(RuntimeError):
             create_setup(devices=s_devices, exptype="BlizzardPQSC")

