import functools
import inspect
import itertools as it
from collections import defaultdict
from collections.abc import Collection
from typing import Type, get_args, get_origin

import numpy as np
import pandas as pd
import rustworkx as rx
from attrs import cmp_using, field
from loguru import logger

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
    measurement_key: str | None = None

    def __call__(self, meas: "MeasurementResult", /, **kwargs) -> "MeasurementResult":
        result = self.run(meas, **kwargs)
        result.processors = (*result.processors, self)
        return result

    def run(self, meas: "MeasurementResult", /, **kwargs) -> "MeasurementResult":
        raise NotImplementedError()


@qdefine
class MeasurementResult:
    """A measurement result object."""

    name: str
    data: pd.DataFrame = field(eq=cmp_using(eq=_dataframe_equals))
    processors: tuple[DataProcessor, ...] = field(factory=tuple)

    def __get__(self, key):
        return self.data[key]

    @property
    def loc(self):
        return self.data.loc


@qdefine
class DataProcessorMetadata:
    """An edge payload for the data processing dependency graph."""

    cls: type[DataProcessor] = field(repr=lambda c: c.__name__)
    before: type[DataProcessor] | None = None
    after: type[DataProcessor] | None = None
    index: int | None = None


GRAPH_IN = True
GRAPH_OUT = False

TMeasurementOrProcessor = (
    type[DataProcessor] | type[MeasurementResult] | type[np.ndarray]
)


@qdefine
class DataProcessorGraph:
    """A data structure for managing the data processing dependency graph."""

    graph: rx.PyDiGraph = field(factory=lambda: rx.PyDiGraph(check_cycle=True))
    registered: dict[TMeasurementOrProcessor, str] = field(factory=dict)
    index_map: dict[str, int] = field(factory=dict)

    @staticmethod
    def get_types_from_signature(
        cls: type[DataProcessor], replace_generic: bool = False
    ) -> tuple[type[MeasurementResult], type[MeasurementResult]]:
        sig = inspect.signature(cls.run)

        in_type = sig.parameters["meas"].annotation
        out_type = sig.return_annotation

        match get_args(in_type):
            case arg, if replace_generic:
                in_type = arg
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
        def decorate(cls: type[DataProcessor]) -> Type[DataProcessor]:
            if not issubclass(cls, DataProcessor):
                raise TypeError(f"'{cls}' is not a subclass of {DataProcessor}.")

            if cls.__name__ in self.index_map:
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
                            raise NotImplementedError()
                        case prev_v:
                            ...

                    self.index_map[name] = (
                        prev_v,
                        new_v := self.graph.add_node(in_type),
                    )

                    self.replace_input_node(prev_v, new_v)

                else:
                    self.registered[name] = in_type
                    self.index_map[name] = (
                        self.graph.add_node(in_type),
                        self.graph.add_node(out_type),
                    )

                v_in, v_out = prev_v, new_v

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

    def get_result_type(
        self, processor_cls: str | type[DataProcessor]
    ) -> type[MeasurementResult]:
        if isinstance(processor_cls, str):
            processor_cls = self.registered[processor_cls]

        _, out_type = DataProcessorGraph.get_types_from_signature(processor_cls)

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
        input_type: type[MeasurementResult] | type[np.ndarray] = np.ndarray,
    ) -> tuple[type[DataProcessor], ...]:
        edge_index = self.index_map[processor_type.__name__]
        _, result_idx, _ = self.graph.edge_index_map()[edge_index]

        input_idx = self.index_map[input_type.__name__]

        try:
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

    key: str
    processor: DataProcessor
    index: int | None = None


@qdefine(eq=False)
class ReadoutPipeline:
    name: str = "default"
    processors: tuple[DataProcessor, ...] = field(factory=tuple)
    dependency_cache: dict[tuple[str, type], MeasurementResult] = field(factory=dict)

    def result_types(self) -> set[Type[MeasurementResult]]:
        """Returns the set of all MeasurementResult subclasses that could be output."""
        return set(DATA_PROCESSORS.get_result_type(p) for p in self.processors)

    @functools.lru_cache
    def get_processor(
        self,
        processor_type: type | str,
        key: str,
    ) -> DataProcessor:
        """Returns a processor with a matching type and compatible measurement key."""
        if not isinstance(processor_type, str):
            processor_type = processor_type.__name__

        processor = None
        for p in self.processors:
            if type(p).__name__ != processor_type:
                continue

            if p.measurement_key == key:
                return p
            elif p.measurement_key is None:
                # We must continue checking if a more specific processor applies
                processor = p

        return processor

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

        processor = self.get_processor(dep, key)
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

            if prev_in != out_result:
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

    def resolve_dependencies(
        self, output_types: dict[str, type[DataProcessor]]
    ) -> list[DataProcessor]:
        def _sort_predecessors(predecessors, keys):
            index = {k: i for i, k in enumerate(keys)}

            return tuple(sorted(predecessors, key=lambda p: index[p[0]]))

        graph = rx.PyDiGraph()
        index_map = dict()

        for key, processor_type in output_types.items():
            dependencies = DATA_PROCESSORS.get_dependencies(processor_type)

            self._build_processor_graph(
                graph, key, prev=None, dependencies=dependencies, index_map=index_map
            )

        resolved_pipeline = []
        for node in rx.topological_sort(graph):
            predecessors = tuple(
                (p.key, type(p.processor)) for p in graph.predecessors(node)
            )

            ## Sort is necessary to match input order to key order.
            if len(predecessors) > 1:
                keys = graph[node].key.split(graph[node].processor.delimiter)
                predecessors = _sort_predecessors(predecessors, keys)

            resolved_pipeline.append(
                (graph[node].key, graph[node].processor, predecessors)
            )

        return resolved_pipeline

    def process_results(
        self,
        input_data: dict[str, MeasurementResult | np.ndarray],
        output_types: dict[str, type[DataProcessor]],
        clear_cache: bool = True,
    ) -> dict[str, MeasurementResult]:
        if clear_cache:
            self.dependency_cache.clear()

        resolved_pipeline = self.resolve_dependencies(output_types=output_types)

        for key, result in input_data.items():
            self.dependency_cache[(key, None)] = result

        for key, processor, predecessors in resolved_pipeline:
            match predecessors:
                case ():
                    inputs = self.dependency_cache[(key, None)]
                case (pred,) if is_generic_type(
                    DATA_PROCESSORS.get_input_type(type(processor)), Collection
                ):
                    inputs = [self.dependency_cache[pred]]
                case (pred,):
                    inputs = self.dependency_cache[pred]
                case (*pred,):
                    inputs = [self.dependency_cache[key] for key in pred]

            self.dependency_cache[(key, type(processor))] = processor(inputs)

        return {
            key: self.dependency_cache[(key, processor_type)]
            for key, processor_type in output_types.items()
        }
