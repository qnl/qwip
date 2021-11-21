from importlib import import_module
from functools import wraps
from typing import Union, Optional, Callable, Any

import attr

from attr import attrib
from loguru import logger

from qwip.parameters import Parameters
from qwip.settings.settings import qattrs, Settings

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

@qattrs
class ProcessSettings(Settings):
    name: str
    process_type: type = attrib(converter=get_process_type)
    inputs: tuple[str, ...] = attrib(factory=tuple)
    parameters: Parameters[str, Any] = attrib(factory=Parameters)

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

@qattrs
class Process:
    settings: ProcessSettings
    completed: bool = False

    def run(self, *inputs):
        raise NotImplementedError('Process subclasses should implement processing logic.')

    @wraps(run)
    def __call__(self, *inputs, **kwargs):
        return self.run(*inputs, **kwargs)

    def update_settings(self):
        to_update = {
            f.name: getattr(self, f.name) for f in attr.fields(type(self)) 
                if f.name not in ('settings', 'completed') and f.metadata.get('serialize', True)
        }
        
        self.settings.parameters.update(to_update)

attr.resolve_types(ProcessSettings)