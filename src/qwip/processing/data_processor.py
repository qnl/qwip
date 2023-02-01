import inspect
from collections import defaultdict
import numpy as np
import pandas as pd
from collections.abc import Collection
from typing import get_origin, get_args

from attrs import field, cmp_using

from qwip.settings.settings import qdefine
from qwip.typing import is_generic_type

DATA_PROCESSOR_LOOKUP = dict()
DATA_PROCESSOR_DEPENDENCIES = dict()

def _dataframe_equals(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    """Checks if two dataframes are equal.

    This function should only ever be called with two DataFrame arguments.

    Returns:
        True if a and b are equal otherwise False.
    """
    return a.equals(b)

@qdefine
class DataProcessor:
    name: str | None = None

    def __call__(self, meas: 'MeasurementResult', /, **kwargs) -> 'MeasurementResult':
        result = self.run(meas, **kwargs)
        result.processors = (*result.processors, self)
        return result

    def run(self, meas: 'MeasurementResult', /, **kwargs) -> 'MeasurementResult':
        raise NotImplementedError()

@qdefine
class MeasurementResult:
    """A measurement result object."""
    name: str
    data: pd.DataFrame = field(
        eq=cmp_using(eq=_dataframe_equals)
    )
    processors: tuple[DataProcessor, ...] = field(factory=tuple)

    def __get__(self, key):
        return self.data[key]

    @property
    def loc(self):
        return self.data.loc


def is_input_output_compatible(out_type, in_type):
    if is_generic_type(in_type, Collection):
        in_type = get_args(in_type)[0]

    return out_type == in_type

def register_data_processor(
    maybe_cls: type[DataProcessor] = None,
    *,
    pre: type['DataProcessor'] | str | None = None,
    post: type['DataProcessor'] | str | None = None,
):
    if pre and not isinstance(pre, str):
        pre = pre.__name__
    
    if post and not isinstance(post, str):
        pre = pre.__name__

    def decorate(cls: type[DataProcessor]) -> type[DataProcessor]:
        if not issubclass(cls, DataProcessor):
            raise TypeError(f'{cls} is not a subclass of {DataProcessor}.')

        sig = inspect.signature(cls.run)

        in_type = sig.parameters['meas'].annotation
        out_type = sig.return_annotation

        DATA_PROCESSOR_LOOKUP[cls.__name__] = (
            cls,
            in_type,
            out_type,
        )

        if cls.__name__ not in DATA_PROCESSOR_DEPENDENCIES:
            DATA_PROCESSOR_DEPENDENCIES[cls.__name__] = []

        if pre:
            if pre not in DATA_PROCESSOR_DEPENDENCIES:
                raise KeyError(
                    f'Pre-processor {pre} is not a registered data processor.'
                )

            _, _, pre_out_type = DATA_PROCESSOR_LOOKUP[pre]
            if not is_input_output_compatible(pre_out_type, in_type):
                raise TypeError(
                    f'Pre-processor {pre} has output type {pre_out_type} that '
                    f'does not match input type {in_type} for {cls.__name__}.'
                )

            DATA_PROCESSOR_DEPENDENCIES[cls.__name__].append(pre)

        if post:
            if post not in DATA_PROCESSOR_DEPENDENCIES:
                raise KeyError(
                    f'Post-processor {pre} is not a registered data processor.'
                )

            _, post_in_type, _ = DATA_PROCESSOR_LOOKUP[post]
            if not is_input_output_compatible(out_type, post_in_type):
                raise TypeError(
                    f'Post-processor {post} has input type {post_in_type} that '
                    f'does not match output type {out_type} for {cls.__name__}.'
                )

            DATA_PROCESSOR_DEPENDENCIES[post].append(cls.__name__)

        return cls

    if maybe_cls:
        return decorate(maybe_cls)

    return decorate

def resolve_class_dependencies(
    name: str
) -> tuple['str']:
    resolved = set()
    visiting = set()

    result = list()

    def add_process(name: str) -> None:
        if name in resolved:
            return
        
        if name in visiting:
            raise ValueError(f'Circular dependency found! {name}')
        
        visiting.add(name)

        for pre in DATA_PROCESSOR_DEPENDENCIES[name]:
            add_process(pre)

        visiting.remove(name)
        resolved.add(name)
        result.append(name)

    add_process(name)

    return tuple(result)

@qdefine
class ReadoutPipeline:
    processors: dict[tuple[str | None, str], DataProcessor] = field(
        factory=dict
    )
    dependency_cache: dict[tuple[str, type], MeasurementResult] = field(
        factory=dict
    )

    def process(
        self,
        meas: dict[str, np.ndarray | MeasurementResult],
        processor_type: type,
        keys: list[str] = None,
        reset_cache: bool = True
    ) -> dict[str, MeasurementResult]:
        results = {}

        for key, inp in meas.items():
            if keys and key not in keys:
                continue
            results[key] = self.process_key(key, inp, processor_type)

        return results

    def get_processor(
        self,
        key: str,
        processor_type: type | str
    ) -> DataProcessor:
        if not isinstance(processor_type, str):
            processor_type = processor_type.__name__
        
        return self.processors.get(
            (key, processor_type),
            self.processors.get((None, processor_type))
        )

    def process_key(
        self,
        key: str,
        data: MeasurementResult | np.ndarray,
        processor_type: str
    ) -> MeasurementResult:
        def _process(key, data, dependencies):
            dep = dependencies.pop()
            dep_cls, in_type, _ =  DATA_PROCESSOR_LOOKUP[dep]

            # Already a processed result
            history = getattr(data, 'processors', tuple())
            if history and isinstance(history[-1], dep_cls):
                return data

            if not (processor := self.get_processor(key, dep_cls)):
                # No matching processor, could be an optional processor
                # so we keep going.
                return _process(key, data, dependencies)

            # Reached root dependency
            if not dependencies:
                # If input datatype doesn't match, raise an error
                if not isinstance(data, in_type):
                    raise TypeError('Wrong type!')

                return processor(data)

            subresult = _process(key, data, dependencies)

            if not isinstance(subresult, in_type):
                raise TypeError('')

            return processor(subresult)
        
        dependencies = list(resolve_class_dependencies(processor_type))

        return _process(key, data, dependencies)
