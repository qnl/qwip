from importlib import import_module
from typing import Optional, Callable, Any, Set

import attr

from attr import field
from loguru import logger

from qwip.flatdict import FlatDict
from qwip.settings.settings import qdefine, Settings
from qwip.processing.process import Process, ProcessSettings

@qdefine
class Pipeline:
    processes: FlatDict[str, Process] = field(factory=FlatDict)
    dependency_cache: dict[str, Any] = field(factory=dict)

    def reset(self):
        self.dependency_cache = dict()
    
    def solve_dependencies(self, name):
        resolved = set()
        visiting = set()

        solved_dependencies = list()

        def add_process(name):
            logger.debug(f'Resolving process {name}.')
            process = self.processes[name]
            if name in resolved:
                return
            if name in visiting:
                logger.error('Circular dependency found!')
                raise Exception

            visiting.add(name)

            for input_name in process.settings.inputs:
                if input_name not in self.processes:
                    logger.error('Dependency not found')
                    raise KeyError(f"'{input_name}' is not in pipeline.processes!")

                add_process(input_name)

            visiting.remove(name)
            resolved.add(name)
            solved_dependencies.append(process)
        
        add_process(name)

        return solved_dependencies
        

    def run(self, input_data, process_name, reset=False):
        processes_to_run = self.solve_dependencies(process_name)

        for p in processes_to_run:
            if not p.settings.inputs:
                output = p(input_data)
            else:
                inputs = tuple(self.dependency_cache[i] for i in p.settings.inputs)
                output = p(*inputs)

            self.dependency_cache[p.settings.name] = output

        return output

    @classmethod
    def from_process_settings(cls, process_list: list[ProcessSettings]):
        return cls(
            processes=FlatDict((p.name, p.get_process()) for p in process_list)
        )

    def to_process_settings(self) -> list[ProcessSettings]:
        psettings = []
        for p in self.processes.values():
            p.update_settings()
            psettings.append(p.settings)

        return psettings

@qdefine
class PipelineSettings(Settings):
    processes: list[ProcessSettings]