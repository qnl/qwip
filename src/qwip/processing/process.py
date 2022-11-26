from importlib import import_module
from functools import wraps
from typing import Union, Optional, Callable, Any

import attr

from attr import attrib
from loguru import logger

import qwip
from qwip.flatdict import FlatDict
from qwip.settings.settings import qdefine, Settings

def get_process_type(process_type: str) -> type:
    """Field converter to get process class from string.

    Args:
        process_type (str): 
    
    Returns:
        (type): The process class specified by process_type.
    """
    if isinstance(process_type, type):
        return process_type

    mod = None
    mod_cls = process_type.rsplit('.', maxsplit=1)
    
    try:
        mod, cls = mod_cls
    except ValueError as e:
        raise ValueError(f'Must specify module to load process type \'{process_type}\' from.') from e

    try:
        mod = import_module(mod)    # import from path
    except ModuleNotFoundError as e1:
        if mod.startswith('qwip.'):
            raise e1

        # allow relative import from qwip.processing
        try:
            mod = import_module(f'.{mod.strip(".")}', package='qwip.processing')
        except ModuleNotFoundError as e2:
            raise ModuleNotFoundError(
                f"No module named '{mod}' or 'qwip.processing.{mod.strip('.')}'"
            )

    cls = getattr(mod, cls)

    return cls

@qdefine
class ProcessSettings(Settings):
    name: str
    process_type: type = attrib(converter=get_process_type, metadata=dict(auto_convert=False))
    inputs: tuple[str, ...] = attrib(factory=tuple)
    parameters: FlatDict[str, Any] = attrib(factory=FlatDict)

    @process_type.validator
    def validate_process_type(self, attribute, value):
        if not issubclass(value, Process):
            raise TypeError(
                f'Process type must be a subclass of \'qwip.processing.process.Process\','
                f' \'{value.__name__}\' is not.'
            )

    def get_process(self, **kwargs) -> 'Process':
        self.parameters.update(**kwargs)
        proc = self.process_type(settings=self, **self.parameters)

        return proc

@qdefine
class Process:
    settings: ProcessSettings = attrib(metadata=dict(serialize=False))
    completed: bool = attrib(default=False, metadata=dict(serialize=False))

    def run(self, *inputs):
        raise NotImplementedError('Process subclasses should implement processing logic.')

    @wraps(run)
    def __call__(self, *inputs, **kwargs):
        return self.run(*inputs, **kwargs)

    def update_settings(self):
        to_update = {}

        for f in attr.fields(type(self)):
            if not f.metadata.get('serialize', True):
                continue

            param = getattr(self, f.name)
            to_update[f.name] = qwip.converter.unstructure(param)
        
        self.settings.parameters.update(to_update)

attr.resolve_types(ProcessSettings)