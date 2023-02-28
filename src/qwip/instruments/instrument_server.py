import importlib

from loguru import logger
from ruamel import yaml

from qwip import qsettings
from qwip.flatdict import FlatDict


class InstrumentServer:
    """ """

    def __init__(self, config_file: str = None, init: bool = False):
        self.instruments = {}

        if config_file:
            self.config = self.load(config_file, init)

    def load(self, config_file: str, init: bool = False):
        with open(config_file, "r") as f:
            config = FlatDict(yaml.load(f, Loader=yaml.CSafeLoader))

        for name, params in config["instruments"].items():
            cls = self.load_driver(params.pop("driver"))
            ins_params = params.pop("parameters", FlatDict())
            self.instruments[name] = cls(name=name, **params)

            if init:
                self.init_instrument(name, ins_params)

        return config

    def init_instrument(self, name: str, parameters: FlatDict):
        ins = self.instruments[name]

        for k, v in parameters.flatitems():
            set_obj = ins

            logger.debug(f"Setting parameter {k} to value: {v}")
            for attribute in parameters.split(k)[:-1]:
                set_obj = getattr(set_obj, attribute)

            attribute = parameters.rsplit(k, maxsplit=1)[-1]
            getattr(set_obj, attribute)(v)

    @staticmethod
    def load_driver(driver: str):
        module, cls = driver.rsplit(".", 1)
        module = importlib.import_module(module)
        cls = module.__getattribute__(cls)

        return cls
