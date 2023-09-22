import importlib
from collections.abc import Mapping
from pathlib import Path

import tomli as tomllib
from attrs import field
from loguru import logger

from qwip.attrs import qdefine


def load_driver(driver: str):
    module, cls = driver.rsplit(".", 1)
    module = importlib.import_module(module)
    cls = module.__getattribute__(cls)

    return cls


@qdefine
class InstrumentServer(Mapping):
    """A central location for managing a collection of instruments."""

    instruments: dict = field(factory=dict)
    config: dict = field(factory=dict, repr=False)

    @classmethod
    def load(cls, config_file: str | Path, init: bool = False):
        """Loads an instrument server from a config file.

        The instrument configuration file should be a toml file with top level keys
        referring to each instrument to be loaded.

        Args:
            config_file: A path to a configuration file.
            init: Whether to initialize the instrument with the given parameters.
        """
        instruments = {}
        with open(config_file, "rb") as f:
            config = tomllib.load(f)

        to_update = {}

        for name, params in config.items():
            ins_cls = load_driver(params["driver"])
            ins_params = params.get("parameters", {})
            init_params = {
                k: v for k, v in params.items() if k not in ("driver", "parameters")
            }
            instruments[name] = ins_cls(name=name, **init_params)

            if init:
                to_update[name] = ins_params

        server = cls(instruments=instruments, config=config)

        for ins, params in to_update.items():
            server.update(ins, params)

        return server

    def update(self, name: str, parameters: dict):
        """Updates an instrument's settings from a parameter dictionary.

        Args:
            name: The instrument to update.
            parameters: A dictionary of parameters.
        """
        ins = self.instruments[name]

        def _set_parameter(obj, key, value):
            match value:
                case dict():
                    for subkey, subvalue in value.items():
                        _set_parameter(getattr(obj, key), subkey, subvalue)
                case _:
                    logger.debug(f"Setting parameter {key} to: {value}")
                    getattr(obj, key)(value)

        for key, value in parameters.items():
            _set_parameter(ins, key, value)

    def __getitem__(self, key: str):
        return self.instruments[key]

    def __iter__(self):
        return self.instruments.__iter__()

    def __len__(self):
        return self.instruments.__len__()

    def close(self):
        """Calls close on all instruments."""
        for ins in self.instruments.values():
            try:
                ins.close()
            except Exception:
                ...
