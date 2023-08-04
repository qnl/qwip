import functools
import inspect
import itertools as it
from collections.abc import Collection
from typing import Any, GenericAlias, TypeVar, get_args, get_origin

import numpy as np
import pandas as pd
import pendulum
import rustworkx as rx
from attrs import cmp_using, field
from loguru import logger
from pendulum import DateTime

import qwip
from qwip._cattr import make_attrs_structure_fn, make_attrs_unstructure_fn
from qwip.attrs import _dataframe_equals, qdefine
from qwip.typing import is_generic_type, issubtype

DATA_PROCESSOR_LOOKUP = dict()
DATA_PROCESSOR_DEPENDENCIES = dict()


def _dataframe_repr(df: pd.DataFrame) -> str:
    """Short representation of DataFrame."""
    rows, cols = df.shape

    return f"DataFrame [{rows} rows x {cols} columns]"


@qdefine
class DataProcessor:
    measurement_key: str | None = None

    def __call__(self, meas: "MeasurementResult", /, **kwargs) -> "MeasurementResult":
        result = self.run(meas, **kwargs)
        match meas:
            case MeasurementResult(timestamp=timestamp):
                ...
            case Collection():
                timestamp = min(m.timestamp for m in meas)

        match result:
            case Collection():
                for res in result:
                    res.processors = (*res.processors, self)
                    res.timestamp = timestamp
            case _:
                result.processors = (*result.processors, self)
                result.timestamp = timestamp

        return result

    def run(self, meas: "MeasurementResult", /, **kwargs) -> "MeasurementResult":
        raise NotImplementedError()

    def output_keys(self) -> set[str]:
        return {self.measurement_key}


@qdefine
class GenericDataProcessor(DataProcessor):
    """A data processor that can act on any measurement result.

    These data processors have no dependencies, so are not added to the dependency
    graph. Instead they are applied at the end, after getting the result from the
    data processor specified as the argument to the class.
    """

    # Using this method for defining a Generic instead of subclassing from Generic
    # because the latter yields a typing._GenericAlias when subscripted instead of
    # a typing.GenericAlias. This makes instance and subclass checks fail.
    # See https://github.com/python/cpython/blob/dbe416b82b8a4ba69d263b915167ea1650ff2412/Lib/_collections_abc.py#L268C5-L268C50
    __class_getitem__ = classmethod(GenericAlias)


@qdefine
class MeasurementResult:
    """A measurement result object.

    Measurement results hold additional metadata on top of a pandas dataframe that
    stores the data. Each result has a name, that should correspond to a measurement
    key.
    """

    name: str
    data: pd.DataFrame = field(
        eq=cmp_using(eq=_dataframe_equals),
        repr=_dataframe_repr,
        metadata=dict(serialize=False),
    )
    timestamp: DateTime = field(factory=pendulum.now, repr=lambda t: t.isoformat())
    processors: tuple[DataProcessor, ...] = field(factory=tuple)

    def __get__(self, key: str) -> Any:
        return self.data[key]

    def __getattr__(self, attr: str) -> Any:
        """Fallback attribute access to the stored dataframe.

        This makes it easy to check attributes on the dataframe. Any attribute that
        also exists on the `MeasurementResult` itself will shadow the dataframe
        attribute, but the dataframe attribute can still be accessed directly like
        `result.data.attribute`.
        """
        return getattr(self.data, attr)

    def _repr_html_(self) -> str:
        description = f'<p style="font-family: monospace;">{repr(self)}</p>'
        dataframe = pd.DataFrame(self.data)._repr_html_()

        return "\n".join([description, dataframe])

    def final_processor(self) -> type[DataProcessor]:
        """Returns the final processor that acted on the result."""

        def get_last_processor(
            processors: tuple[DataProcessor, ...]
        ) -> type[DataProcessor] | None:
            """Recursive lookup of last processor, accounting for generics."""
            try:
                fp = processors[-1]
                if isinstance(fp, GenericDataProcessor):
                    np = get_last_processor(processors[:-1])
                    return type(fp) if np is None else type(fp)[np]
                else:
                    return type(fp)

            except IndexError:
                return None

        return get_last_processor(self.processors)


@qdefine
class DataProcessorMetadata:
    """An edge payload for the data processing dependency graph."""

    cls: type[DataProcessor] = field(repr=lambda c: c.__name__)
    before: type[DataProcessor] | None = None
    after: type[DataProcessor] | None = None
    index: int | None = None


GRAPH_IN = True
GRAPH_OUT = False

TMeasurementOrProcessor = type[DataProcessor] | type[MeasurementResult]


@qdefine
class DataProcessorGraph:
    """A data structure for managing the data processing dependency graph."""

    graph: rx.PyDiGraph = field(factory=lambda: rx.PyDiGraph(check_cycle=True))
    registered: dict[str, TMeasurementOrProcessor] = field(factory=dict)
    index_map: dict[str, int] = field(factory=dict)

    @staticmethod
    def get_types_from_signature(
        cls: type[DataProcessor], replace_generic: bool = False
    ) -> tuple[type[MeasurementResult], type[MeasurementResult]]:
        sig = inspect.signature(cls.run)

        in_type = sig.parameters["result"].annotation
        out_type = sig.return_annotation

        match get_args(in_type):
            case arg, if replace_generic:
                in_type = arg
            case _:
                ...

        match get_args(out_type):
            case arg, if replace_generic:
                out_type = arg
            case _:
                ...

        return in_type, out_type

    def replace_input_node(self, original: int, new: int) -> None:
        """Replaces the original input node for all associated edges.

        Original:
            original =edge=> v_out

        New:
            new =edge=> v_out
        """

        v_outs = self.graph.adj_direction(original, GRAPH_OUT)

        for v_out, edge in v_outs:
            self.graph.remove_edge(original, v_out)
            index = self.graph.add_edge(new, v_out, edge)
            self.index_map[edge.cls.__name__] = index
            edge.index = index

    def replace_output_node(self, original: int, new: int) -> None:
        """Replaces the original output node for all associated edges.

        Original: v_in =edge=> original
        New: v_in =edge=> new

        Args:
            original: The graph node index of the original output node.
            new: The graph node index of the new output node
        """

        v_ins = self.graph.adj_direction(original, GRAPH_IN)

        for v_in, edge in v_ins:
            self.graph.remove_edge(v_in, original)
            index = self.graph.add_edge(v_in, new, edge)
            self.index_map[edge.cls.__name__] = index
            edge.index = index

    def register(
        self,
        maybe_cls: type[DataProcessor] | None = None,
        *,
        before: type[DataProcessor] | None = None,
        after: type[DataProcessor] | None = None,
    ):
        def decorate(cls: type[DataProcessor]) -> type[DataProcessor]:
            if not issubclass(cls, DataProcessor):
                raise TypeError(f"'{cls}' is not a subclass of {DataProcessor}.")

            if cls.__name__ in self.registered:
                return cls

            if issubclass(cls, GenericDataProcessor):
                self.registered[cls.__name__] = cls
                return cls

            in_type, out_type = DataProcessorGraph.get_types_from_signature(
                cls, replace_generic=True
            )

            # We need to unroll the loop to preserve the DAG if the processor
            # outputs the same type as it takes in.
            if in_type == out_type:
                name = in_type.__name__
                if name in self.index_map:
                    match self.index_map[name]:
                        case tuple(nodes):
                            # We need a way to resolve the ambiguity here. Since it is
                            # no longer possible to determine where the node should go
                            raise NotImplementedError()
                        case prev_v:
                            ...

                    self.index_map[name] = (
                        prev_v,
                        new_v := self.graph.add_node(in_type),
                    )

                    self.replace_input_node(prev_v, new_v)
                    v_in, v_out = prev_v, new_v

                else:
                    self.registered[name] = in_type
                    v_in, v_out = self.index_map[name] = (
                        self.graph.add_node(in_type),
                        self.graph.add_node(out_type),
                    )

            else:
                # If processor connects two different result types
                if in_type.__name__ not in self.index_map:
                    self.registered[in_type.__name__] = in_type
                    self.index_map[in_type.__name__] = self.graph.add_node(in_type)

                if out_type.__name__ not in self.index_map:
                    self.registered[out_type.__name__] = out_type
                    self.index_map[out_type.__name__] = self.graph.add_node(out_type)

                match self.index_map[in_type.__name__]:
                    case (*vs, v_in):
                        ...
                    case v_in:
                        ...

                match self.index_map[out_type.__name__]:
                    case (v_out, *vs):
                        ...
                    case v_out:
                        ...

            self.registered[cls.__name__] = cls
            index = self.index_map[cls.__name__] = self.graph.add_edge(
                v_in, v_out, DataProcessorMetadata(cls=cls, before=before, after=after)
            )
            _, _, metadata = self.graph.edge_index_map()[index]
            metadata.index = index

            return cls

        if maybe_cls:
            return decorate(maybe_cls)

        return decorate

    def get_root_node(self) -> type[MeasurementResult]:
        """Returns the root node of the processor graph.

        Raises:
            ValueError: If more than one root node exists for the graph, or if the graph
                is empty.
        """
        visited = set()

        nodes = self.graph.nodes()
        root = None

        while nodes:
            root = nodes[0]
            idx = self.index_map[root.__name__]

            if idx in visited:
                raise ValueError(
                    "DataProcessorGraph contains a cycle. This should not occur."
                )

            if isinstance(idx, tuple):
                idx, *rest = idx

            nodes = self.graph.predecessors(idx)
            visited.add(idx)

        if root is None:
            raise ValueError("Cnanot get root node of an empty graph.")

        return root

    def get_result_type(
        self, processor_cls: str | type[DataProcessor], replace_generic: bool = False
    ) -> type[MeasurementResult]:
        if isinstance(processor_cls, str):
            processor_cls = self.registered[processor_cls]

        _, out_type = DataProcessorGraph.get_types_from_signature(
            processor_cls, replace_generic
        )

        return out_type

    def get_input_type(
        self, processor_cls: str | type[DataProcessor], replace_generic: bool = False
    ) -> type[MeasurementResult]:
        if isinstance(processor_cls, str):
            processor_cls = self.registered[processor_cls]

        in_type, _ = DataProcessorGraph.get_types_from_signature(
            processor_cls, replace_generic
        )

        return in_type

    def get_dependencies(
        self,
        processor_type: type[DataProcessor],
        input_type: type[MeasurementResult] | None = None,
    ) -> tuple[type[DataProcessor], ...]:
        edge_index = self.index_map[processor_type.__name__]
        _, result_idx, _ = self.graph.edge_index_map()[edge_index]

        input_type = input_type or self.get_root_node()
        input_idx = self.index_map[input_type.__name__]

        try:
            match input_idx:
                # We take the last input_idx b.c. it's supposed to be the shortest path
                # May want to consider resolving the ambiguity in a more flexible way
                case (*prev, input_idx):
                    ...

            path = rx.dijkstra_shortest_paths(
                self.graph,
                input_idx,
                target=result_idx,
            )[result_idx]
        except IndexError:
            raise ValueError(
                f"Path from input '{input_type.__name__}' to final data processor "
                f"'{processor_type.__name__}' does not exist."
            )

        return [self.graph.get_edge_data(i, o).cls for i, o in it.pairwise(path)]

    def measurement_results(self) -> set[MeasurementResult]:
        """Returns the set of all MeasurementResult classes that have been registered."""
        return set(
            cls
            for cls in self.registered.values()
            if issubclass(cls, MeasurementResult | np.ndarray)
        )

    def data_processors(self) -> set[DataProcessor]:
        """Returns the set of all DataProcessor classes that have been registered."""

        return set(
            cls for cls in self.registered.values() if issubclass(cls, DataProcessor)
        )


DATA_PROCESSORS = DataProcessorGraph()


@qdefine
class KeyProcessorNode:
    """Payload data for a graph node when building out the dependency graph."""

    key: str | None
    processor: DataProcessor
    index: int | None = None


@qdefine(eq=False)
class ReadoutPipeline:
    name: str = "default"
    processors: tuple[DataProcessor, ...] = field(factory=tuple)
    dependency_cache: dict[
        tuple[str, type[DataProcessor] | None], MeasurementResult
    ] = field(factory=dict)

    def result_types(self) -> set[type[MeasurementResult]]:
        """Returns the set of all MeasurementResult subclasses that could be output."""
        return set(
            DATA_PROCESSORS.get_result_type(p, replace_generic=True)
            for p in self.processors
        )

    @functools.lru_cache
    def get_processor(
        self,
        processor_type: type[DataProcessor] | str,
        input_key: str | None = None,
        output_key: str | None = None,
    ) -> DataProcessor | None:
        """Returns a processor with a matching type and compatible measurement key.

        Args:
            processor_type: A `DataProcessor` subclass or a string specifying a subclass.
            input_key: A measurement key. If `None`, any measurement key will match.
            output_key: An output meausurement key, if `None` any output key will match.

        Returns:
            A compatible data processor, or `None` if no data processor is found.
        """
        if not isinstance(processor_type, str):
            processor_type = processor_type.__name__

        processor = None
        for p in self.processors:
            if type(p).__name__ != processor_type:
                continue

            output_keys = p.output_keys()

            if (p.measurement_key == input_key or input_key is None) and (
                output_key in output_keys or output_key is None
            ):
                return p

            # In these cases we must check if a more specific processor matches.
            input_fallback_match = (
                p.measurement_key == input_key
                or input_key is None
                or p.measurement_key is None
            )
            output_fallback_match = (
                output_key in output_keys or output_key is None or None in output_keys
            )

            if input_fallback_match and output_fallback_match:
                processor = p

        return processor

    def add_processor(self, processor: DataProcessor) -> None:
        """Adds a processor to the pipeline.

        If a processor with matching type and key already exists, it will be replaced.

        Args:
            processor: The DataProcessor to be added.
        """

        self.get_processor.cache_clear()

        def should_remove(other):
            return (
                type(other) == type(processor)
                and other.measurement_key == processor.measurement_key
            )

        self.processors = (
            *(proc for proc in self.processors if not should_remove(proc)),
            processor,
        )

    def remove_processor(
        self,
        processor_type: type[DataProcessor] | str,
        key: str | None = None,
    ) -> DataProcessor:
        """Removes a processor from the pipeline.

        Unlike add_processor, the processor_type and key must match exactly, i.e. a
        processor with `measurement_key == None` will not be removed if key is given
        and not `None`.

        Args:
            processor_type: A `DataProcessor` subclass or a string specifying a subclass.
            key: A measurement key.

        Returns:
            The `DataProcessor` that was removed.

        Raises:
            KeyError: If the specified processor does not exist.
        """

        if not isinstance(processor_type, str):
            processor_type = processor_type.__name__

        new_processors = []

        removed = None
        for p in self.processors:
            if type(p).__name__ == processor_type and p.measurement_key == key:
                removed = p
            else:
                new_processors.append(p)

        if removed is None:
            raise KeyError(
                f"DataProcessor of type {processor_type} with measurement key {key} "
                f"does not exist."
            )

        self.processors = new_processors
        self.get_processor.cache_clear()

        return removed

    def _build_processor_graph(
        self,
        graph: rx.PyDiGraph,
        key: str,
        prev: int | None = None,
        dependencies=[],
        index_map={},
    ) -> None:
        """A helper function for building the processing chain."""
        if not dependencies:
            return

        dep = dependencies.pop()

        in_result = DATA_PROCESSORS.get_input_type(dep)
        out_result = DATA_PROCESSORS.get_result_type(dep)

        # Need to handle processors that return a list of results
        if is_generic_type(out_result, Collection):
            processor = self.get_processor(dep, None, key)
            if processor:
                key = processor.measurement_key
        else:
            processor = self.get_processor(dep, key, None)

        if not processor:
            # No matching processor, so we skip b/c it could be an optional dependency
            return self._build_processor_graph(
                graph, key, prev, dependencies, index_map
            )

        if (key, type(processor)) in index_map:
            index = index_map[(key, type(processor))]
        else:
            logger.debug(f"Adding node ({key}, {type(processor).__name__})")
            index = graph.add_node(KeyProcessorNode(key=key, processor=processor))
            graph[index].index = index
            index_map[(key, type(processor))] = index

        out_node = graph[index]

        if prev:
            prev_in = DATA_PROCESSORS.get_input_type(
                type(prev.processor), replace_generic=True
            )
            curr_out = DATA_PROCESSORS.get_result_type(
                type(processor), replace_generic=True
            )

            if prev_in != curr_out and TypeVar not in (type(prev_in), type(curr_out)):
                raise ValueError(
                    f"Input result type for {type(prev.processor).__name__} "
                    f"does not match output result type for {dep.__name__ }. "
                    f"This is likely due to a missing processor for key '{key}''."
                )

            graph.add_edge(out_node.index, prev.index, None)

        if is_generic_type(in_result):
            next_keys = key.split(processor.delimiter)
        else:
            next_keys = [key]

        for key in next_keys:
            self._build_processor_graph(
                graph,
                key=key,
                prev=out_node,
                dependencies=[d for d in dependencies],
                index_map=index_map,
            )

    def dependency_graph(
        self, output_types: dict[str, type[DataProcessor] | None]
    ) -> rx.PyDiGraph:
        """Builds the full dependency graph for the processing pipeline.

        Args:
            output_types: The desired final processor for each measurement key.

        Returns:
            A directed graph with `KeyProcessorNode` as the node payload. Each node
            holds the measurement key and a data processor. A directed edge from node
            A to node B implies that node B depeneds on the result from node A, and thus
            must be run first.
        """
        graph = rx.PyDiGraph()
        index_map = dict()

        for key, processor_type in output_types.items():
            generics = []
            while issubtype(processor_type, GenericDataProcessor):
                generics.append(get_origin(processor_type) or processor_type)

                match get_args(processor_type):
                    case ():
                        processor_type = None
                    case (processor_type,):
                        ...
                    case _:
                        ValueError(
                            f"GenericDataProcessors should only have a single argument,"
                            f" got {processor_type}."
                        )

            if processor_type is None:
                dependencies = [] + generics[::-1]
            else:
                dependencies = (
                    DATA_PROCESSORS.get_dependencies(processor_type) + generics[::-1]
                )

            self._build_processor_graph(
                graph, key, prev=None, dependencies=dependencies, index_map=index_map
            )

        return graph

    def resolve_dependencies(
        self, output_types: dict[str, type[DataProcessor]]
    ) -> list[DataProcessor]:
        """Returns a toplogical ordering of measurement key/processor pairs.

        Args:
            output_types: The desired final processor for each measurement key.

        Returns:
            A list of `(measurement_key, processor, predecessor)` tuples. The list is a
            topological sort of the dependency graph such that all dependencies for a
            given processor always come before that processor.
        """

        def _sort_predecessors(predecessors, keys):
            index = {k: i for i, k in enumerate(keys)}

            return tuple(sorted(predecessors, key=lambda p: index[p[0]]))

        graph = self.dependency_graph(output_types)

        resolved_pipeline = []
        for node in rx.topological_sort(graph):
            node_key = graph[node].key
            predecessors = tuple(
                (
                    # Need to handle processors with multiple output keys
                    node_key if len(p.processor.output_keys()) > 1 else p.key,
                    type(p.processor),
                )
                for p in graph.predecessors(node)
            )

            ## Sort is necessary to match input order to key order.
            if len(predecessors) > 1:
                keys = graph[node].key.split(graph[node].processor.delimiter)
                predecessors = _sort_predecessors(predecessors, keys)

            resolved_pipeline.append(
                (graph[node].key, graph[node].processor, predecessors)
            )

        return resolved_pipeline

    def _get_inputs(
        self, key: str, previous: type[DataProcessor] | None, processor: DataProcessor
    ) -> MeasurementResult | None:
        """Helper method for getting an input from the dependency cache.

        This method will look for a proper input result to the current processor in the
        dependency cache. It will first look for a matching key and a matching previous
        processor. A previous processor type of `None` is used to signify that the
        result was passed in by the user. If no matching key and processor is found,
        it will then look for a matching key and a matching input type. If there is
        still no suitable input, then it will return `None`.

        Args:
            key: The current measurement key.
            previous: The processor type that produced the previous output. `None` is
                used to specify that the result was passed in by the user.
            processor: The current processor that will process the input results.

        Returns:
            The requested input, or `None` if no input was found.
        """
        expected_input = DATA_PROCESSORS.get_input_type(
            type(processor), replace_generic=True
        )

        try:
            inputs = self.dependency_cache[(key, previous)]
            # This is necessary for handling the case where previous is None
            if isinstance(expected_input, TypeVar) or isinstance(
                inputs, expected_input
            ):
                return inputs
        except KeyError:
            pass

        inputs = self.dependency_cache.get((key, None))
        if isinstance(expected_input, TypeVar) or isinstance(inputs, expected_input):
            return inputs

        return None

    def process_results(
        self,
        input_data: dict[str, MeasurementResult],
        output_types: dict[str, type[DataProcessor]],
        clear_cache: bool = True,
        **kwargs,
    ) -> dict[str, MeasurementResult]:
        """Processes the input data.

        Args:
            input_data: A mapping of measurement keys to input data.
            output_types: The desired final processor for each measurement key.
            clear_cache: If `True` the dependency cache is cleared before processing
                the results.

        Returns:
            A mapping of measurement keys to results.

        Raises:
            KeyError: If any dependencies are missing in the given input data.
        """
        if clear_cache:
            self.dependency_cache.clear()

        resolved_pipeline = self.resolve_dependencies(output_types=output_types)

        for key, result in input_data.items():
            self.dependency_cache[(key, None)] = result

        for key, processor, predecessors in resolved_pipeline:
            match predecessors:
                case ():
                    inputs = self._get_inputs(key, None, processor)
                case (*pred,) if is_generic_type(
                    DATA_PROCESSORS.get_input_type(type(processor)), Collection
                ):
                    inputs = [self._get_inputs(*pair, processor) for pair in pred]
                    if None in inputs:
                        inputs = None
                case (pred,):
                    _, prev_processor = pred
                    inputs = self._get_inputs(key, prev_processor, processor)

            if inputs is None:
                logger.debug(
                    f"Skipping {type(processor).__name__} for key {key} due to missing input."
                )
                continue

            result = processor(inputs, **kwargs)
            if isinstance(result, Collection):
                self.dependency_cache.update(
                    {(res.name, type(processor)): res for res in result}
                )
            else:
                self.dependency_cache[(key, type(processor))] = result

        return {
            key: self.dependency_cache[
                (key, get_origin(processor_type) or processor_type)
            ]
            for key, processor_type in output_types.items()
        }

    def cached_data(
        self,
        processor: type[DataProcessor] | None = None,
        keys: list[str] | None = None,
    ) -> dict[str, MeasurementResult]:
        """Returns measurement results stored in the dependency cache.

        This is useful for inspecting the partially processed data. For example, looking
        at the raw IQ data points or the classified data on a per shot basis, without
        rerunning the processing pipeline.

        Args:
            processor: The final processor class of the requested data.
            keys: A list of measurement keys to get data for. If `None` all results that
                match the processor will be returned.

        Returns:
            A mapping of measurement keys to the requested results.
        """

        def filter_func(key: str, proc: type[DataProcessor]) -> bool:
            return (keys is None or key in keys) and (proc == processor)

        results = {
            key: data
            for (key, proc), data in self.dependency_cache.items()
            if filter_func(key, proc)
        }

        return results


# ========== Data Processor converters ========== #


def make_data_processor_structure_fn(cls):
    structure_attrs = make_attrs_structure_fn(cls)

    def structure_fn(val, cls):
        if isinstance(val, cls):
            return val

        subclass = DATA_PROCESSORS.registered.get(
            val.get("__class__", None),
        )

        if subclass is None:
            logger.warning(
                f"No matching data processor found. Structuring {val} as {cls}."
            )
            return structure_attrs(val, cls)

        return qwip.converter.structure(val, subclass)

    return structure_fn


def make_data_processor_unstructure_fn(cls):
    unstructure_attrs = make_attrs_unstructure_fn(cls, omit_defaults=False)

    def unstructure_fn(obj):
        return {**unstructure_attrs(obj), "__class__": type(obj).__name__}

    return unstructure_fn


qwip.converter.register_structure_hook_factory(
    lambda cls: cls in (DataProcessor, MeasurementResult),
    make_data_processor_structure_fn,
)

qwip.converter.register_unstructure_hook_factory(
    lambda cls: issubclass(cls, (DataProcessor, MeasurementResult)),
    make_data_processor_unstructure_fn,
)
