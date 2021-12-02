"""Processing blocks that perform useful utility functions."""
import re
from typing import Any, Callable, Mapping, Union

import numpy as np
from attr import attrib
from attr.validators import in_

from qwip.flatdict import FlatDict
from qwip.settings.settings import qattrs
from qwip.processing.process import Process

@qattrs
class FormatLegacyHeterodyne(Process):
    """A processing block to reformat a legacy QTRL `meas` dictionary.

    This process will pull all keys matching the regex 'R(\\d+)' and reshape
    the IQ data from (IQ, ...) to (..., IQ) because this makes more sense.
    """

    def run(self, meas, /):
        """Returns a data dictionary from a QTRL `meas` dictionary.
        
        Args:
            meas (dict): A QTRL `meas` dictionary.

        Returns:
            (dict): A dictionary mapping resonator keys to reshaped IQ data
        """
        output = {}

        for key, res in meas.items():
            if not re.fullmatch('R\\d+', key):
                continue

            data = res['Heterodyne']

            output[key] = np.array(np.moveaxis(data, 0, -1), order='C')
        
        return output

@qattrs
class Rename(Process):
    """A processing block to rename a data key.
    
    The renaming can be specified with a mapping that takes old keys to new keys
    """
    def _create_rename_func(maybe_mapping: Union[Callable, Mapping]):
        if isinstance(maybe_mapping, Mapping):
            return lambda x: maybe_mapping.get(x, x)
        else:
            return maybe_mapping

    rename: Callable[[str], str] = attrib(converter=_create_rename_func, metadata=dict(auto_convert=False))

    def run(self, data, /):
        """Renames the data passed to subsequent blocks."""

        return {self.rename(k): v for k, v in data.items()} 


@qattrs
class FilterData(Process):
    """A processing block to filter data passed to subsequent blocks.
    
    The filter can be specified with a regex, a list of allowed keys, or a
    callable that returns a boolean.
    """

    def _filter_func_from_condition(cond):
        if isinstance(cond, str):
            return lambda x: bool(re.fullmatch(cond, x))
        elif hasattr(cond, '__contains__'):
            return lambda x: x in cond
        else:
            return cond

    filter: Callable[..., bool] = attrib(
        converter=_filter_func_from_condition,
        metadata=dict(auto_convert=False)
    )

    def run(self, data, /):
        """Filters the data passed to subsequent blocks."""

        return {k: v for k, v in data.items() if self.filter(k)}

@qattrs
class CollectData(Process):
    """A processing block to collect outputs from multiple input blocks.
    
    The results are placed in a `FlatDict` object where the key corresponds
    to the name of the input process.
    """
    outer_key: str = attrib(default='process',
                            validator=in_(('process', 'label')))

    def run(self, *data):
        if self.outer_key == 'process':
            return FlatDict({k: d for k, d in zip(self.settings.inputs, data)})
        else:
            output = FlatDict()
            for proc_key, proc_data in zip(self.settings.inputs, data):
                if isinstance(proc_data, Mapping):
                    for label in proc_data:
                        if label not in output:
                            output[label] = FlatDict()

                        output[label][proc_key] = proc_data[label]

                else:
                    output[proc_key] = proc_data        
            return output