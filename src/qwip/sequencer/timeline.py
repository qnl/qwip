import itertools as it
from collections.abc import Callable, Collection, Iterable, Iterator
from copy import copy, deepcopy
from functools import singledispatchmethod
from numbers import Real
from typing import Literal, Self

import matplotlib.pyplot as plt
import numpy as np
import sympy as sym
from attrs import field
from loguru import logger
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import qwip
from qwip._cattr import make_attrs_structure_fn
from qwip.attrs import qdefine
from qwip.sequencer.utils import _variable_substitution
from qwip.sequencer.waveform import (
    CosineRampWaveform,
    InfiniteWaveform,
    Marker,
    Operation,
    Waveform,
)
from qwip.utils import deprecated
from qwip.visualization.utils import all_legend_handles_labels

Location = sym.Expr
LocationLike = Location | str | Real
TChannelMap = dict[str, tuple[Location, Operation]]


def _default_sort_key(loc_op: tuple[float, Operation]) -> float:
    return loc_op[0]


@qdefine
class Timeline:
    lw_pairs: list[tuple[Location, Operation]] = field(factory=list)
    width: Location | None = None
    constraints: set[sym.Expr] = field(factory=set)
    channels: set[str] = field(factory=set, metadata=dict(serialize=False))

    def __attrs_post_init__(self):
        # Update channels from waveforms
        self.channels.update(
            op.channel
            for op in self.operations
            if hasattr(op, "channel") and op.channel
        )

    @property
    def locations(self) -> list[Location]:
        return [loc for loc, _ in self]

    @property
    def operations(self) -> list[Operation]:
        return [op for _, op in self]

    @classmethod
    def fromtuples(
        cls,
        locations: list[tuple[LocationLike, Operation | None]],
        width: LocationLike | None = None,
        constraints: list[sym.Expr] = [],
    ) -> Self:
        """Constructs a sequence from a tuple of locations and waveforms.

        Args:
            locations: A list of (location, pulse) pairs to add to the sequence
            width: The width of timeline.
            constraints: A list of sympy expressions specifying constraints on the
                variables.

        Returns:
            The resulting `Timeline` instance
        """

        return cls(lw_pairs=locations, width=width, constraints=constraints)

    @classmethod
    def from_dict(
        cls,
        pulse_locations: dict[LocationLike, list[Operation]],
        width: LocationLike | None = None,
        constraints: list[sym.Expr] = [],
    ) -> Self:
        locations = []
        for loc, waves in pulse_locations.items():
            for w in waves:
                loc = qwip.converter.structure(loc, sym.Expr)
                locations.append((loc, w))

        return cls(lw_pairs=locations, width=width, constraints=constraints)

    @classmethod
    def from_layers(cls, layers: list, t0: LocationLike = 0, **kwargs) -> Self:
        """Creates a timeline from a list of circuit/gate layers.

        Each layer consists of one or more delay times or operations that are meant to
        be played at the same start time. The start time of each subsequent layers is
        determined by finding the maximum duration of all operations and adding it to
        the start time of the previous layer.

        All operations in the same layer are left justified to the start time of that
        layer.

        Args:
            layers: A list of layers.
            t0: The initial start time of the first layer.
            **kwargs: Remaining keyword arguments are passed to the `Timeline.__init__`
                method.

        Raises:
            ValueError: if the duration of a given layer cannot be determined
                unambigously. This can occur when there is only a single timeline with
                no width in a given layer, or if there are two variable widths in the
                same layer.
        """
        tmln = cls(**kwargs)

        def get_layer_width(widths: set[Location | None]) -> Location:
            if not widths:
                return 0

            widths.discard(None)

            if len(widths) == 0:
                raise ValueError(f"Layer with width {None} has ambiguous timing.")

            elif len(widths) == 1:
                return widths.pop()

            try:
                return max(widths)
            except ValueError as e:
                raise ValueError(f"Layer with widths {widths} has ambiguous timing.")

        t0 = qwip.converter.structure(t0, sym.Expr)

        for layer in layers:
            widths = set()

            if isinstance(layer, (str, Timeline)) or not isinstance(layer, Iterable):
                layer = [layer]

            for op in layer:
                match op:
                    case Timeline():
                        tmln.add(op, t0)
                        width = op.width
                    case Operation():
                        tmln.add(op, t0)
                        width = op.width
                    case Real():
                        width = op
                    case str():
                        width = op
                    case Location():
                        width = op
                    case None:
                        continue
                    case _:
                        raise ValueError(f"{op} is not a recognized operation.")

                widths.add(qwip.converter.structure(width, sym.Expr))

            t0 += get_layer_width(widths)

        return tmln

    def add(self, target: Self | Operation, /, location: LocationLike = 0.0) -> Self:
        """Adds a waveform or another pulse timeline to the specified location.

        Args:
            target: The waveform, list of waveforms or other pulse timeline to add.
            location: The location in the current pulse timeline at which the target
                should be added.

        Returns:
            The current pulse timeline.
        """

        match target:
            case Timeline():
                return self.add_timeline(target, self_loc=location)
            case _:
                return self.add_waveform(target, location=location)

    def _add_location_waveform_pair(self, loc: LocationLike, wave: Operation) -> None:
        loc = qwip.converter.structure(loc, sym.Expr)
        self.lw_pairs.append((loc, wave))

    def add_waveform(
        self,
        waveforms: Operation | Iterable[Operation],
        location: LocationLike = 0.0,
    ) -> Self:
        """Adds a waveform to the pulse timeline at the specified location.

        Args:
            location: The location at which to place the waveform
            waveforms: The waveform to add
        """
        if isinstance(waveforms, Operation):
            waveforms = [waveforms]

        for wave in waveforms:
            self._add_location_waveform_pair(location, wave)

        self.channels.update(wave.channel for wave in waveforms if wave.channel)

        return self

    def remove_waveform(
        self,
        waveform: Operation | Collection[Operation],
        location: LocationLike | None = None,
    ) -> Self:
        """Removes a waveform to the pulse timeline.

        By default will remove all instances of the waveform(s). If a
        location is specified, only instances of the waveform at the
        specified location are removed.
        """

        if isinstance(waveform, Operation):
            waveform = (waveform,)

        if location is not None:
            if not isinstance(location, Location):
                location = Location(location)

            self.locations[location][:] = [
                w for w in self.locations[location] if w not in waveform
            ]

            return self

        for loc in self.locations:
            self.locations[loc][:] = [
                w for w in self.locations[loc] if w not in waveform
            ]

        return self

    def add_constraints(
        self,
        *constraints: Real | str | sym.Expr,
        **substitutions: Real | str | sym.Expr,
    ) -> None:
        """Adds constraints to the set of existing constraints.

        This function takes in a list of constraints that are converted to sympy
        expressions or a keyword mapping `var_name = expr` that is converted to a
        constraint of the form `var_name - expr = 0`.

        Args:
            *constraints: Positional constraints are converted directly to sympy
                expressions.
            **substitutions: Keyword constraints are specified as variable substitutions
                of the form `var_name = expr`.
        """

        self.constraints.update(qwip.converter.structure(constraints, list[sym.Expr]))
        self.constraints.update(
            {
                sym.Symbol(symbol) - qwip.converter.structure(value, sym.Expr)
                for symbol, value in substitutions.items()
            }
        )

    def remove_constraint(self, constraint: Real | str | sym.Expr) -> sym.Expr | None:
        """Removes a constraint from the constraint mapping.

        Args:
            constraint: The constraint to remove.

        Returns:
            The constraint that was removed or `None` if it was not found.
        """

        constraint = qwip.converter.structure(constraint, sym.Expr)

        if constraint not in self.constraints:
            return None

        self.constraints.remove(constraint)
        return constraint

    def add_timeline(
        self,
        other: Self,
        self_loc: LocationLike = 0.0,
        other_loc: LocationLike = 0.0,
        name: str | None = None,
    ) -> Self:
        """Adds another pulse timeline to the current timeline.

        With default arguments, this is equivalent to:

        >>> self += other

        However, this method enables other to be translated on the fly by passing in
        locations in each timeline to line up.

        Args:
            other: The pulse timeline to append.
            self_loc: The location in the current pulse timeline to line up with
                the location in the `other` pulse timeline.
            other_loc: The location in the `other` pulse timeline to line up with
                the location in the current pulse timeline.
            name: A variable name to define as the new location of the origin
                for the `other` sequence. This makes it simpler to shift the origin
                later.

        Raises:
            ValueError: If any constraints that are declared in both sequence
                elements and differ from each other.
        """
        if not isinstance(self_loc, Location):
            self_loc = qwip.converter.structure(self_loc, sym.Expr)

        if not isinstance(other_loc, Location):
            other_loc = qwip.converter.structure(other_loc, sym.Expr)

        dt = self_loc - other_loc
        if name is not None:
            name = sym.Symbol(name)
            self.constraints.add(name - dt)
            dt = name

        for loc, op in other:
            loc += dt
            self.lw_pairs.append((loc, op))

        self.constraints.update(other.constraints)

        self.channels.update(other.channels)

        return self

    def variables(self) -> set[str]:
        """Returns the set of variables referenced in the pulse timeline.

        Returns:
            A set that contains the names of all variables referenced within the
            location mapping or the constraint mapping.
        """
        symbols = set().union(
            *(expr.free_symbols for expr in it.chain(self.locations, self.constraints)),
            set() if self.width is None else self.width.free_symbols,
        )

        tvars = {s.name for s in symbols}
        opvars = set().union(*(op.variables() for op in self.operations))

        return tvars | opvars

    def rename_variables(self, rename_func: Callable[[str], str]) -> dict[str, str]:
        """Renames all variables according to a renaming function.

        Use `Timeline.substitute` to replace variables via a mapping of values.

        Args:
            rename_func: a function that, given a string, returns a new string.

        Returns:
            A set containing the new names of all variables referenced by the
            pulse timeline.
        """
        var_map = {
            old: new for old in self.variables() if old != (new := rename_func(old))
        }

        if not var_map:
            return var_map

        self.lw_pairs[:] = (
            (_variable_substitution(loc, var_map), op.resolve(**var_map))
            for loc, op in self
        )

        constraints = [_variable_substitution(c, var_map) for c in self.constraints]
        self.constraints.clear()
        self.constraints.update(constraints)

        if self.width:
            self.width = _variable_substitution(self.width, var_map)

        return var_map

    def substitute(self, **substitutions: Real | str | sym.Expr) -> Self:
        """Substitutes new values for a set of variables.

        This method modifies the timeline in place.

        Args:
            **substitutions: Should be a mapping from variable names to their values.
                These values can be either an expression or a numerical value.

        Returns:
            The modified timeline.
        """
        var_set = set(substitutions)

        lw_pairs = []
        for loc, op in self:
            if {s.name for s in loc.free_symbols} & var_set:
                loc = _variable_substitution(loc, substitutions)

            if op.variables() & var_set:
                op = op.resolve(**substitutions)

            lw_pairs.append((loc, op))

        if self.width and var_set & {s.name for s in self.width.free_symbols}:
            self.width = _variable_substitution(self.width, substitutions)

        if self.constraints:
            new_constraints = {
                _variable_substitution(cons, substitutions) for cons in self.constraints
            }
            self.constraints.clear()
            self.constraints.update(new_constraints)

        self.lw_pairs[:] = lw_pairs

        return self

    def solve_constraints(
        self, *constraints: Real | str | sym.Expr
    ) -> dict[str, Location]:
        """Solves all timing constraints for the pulse timeline.

        **kwargs: Keyword arguments can be used to add constraints and are passed
            directly to `add_constraints`.

        Returns:
            A dictionary mapping all location names to concrete locations.
        """
        constraints = list(self.constraints | set(constraints))
        # Needed to deal with variable names that include .
        variables = {sym.Symbol(v) for v in self.variables()}

        if not variables:
            return {}

        sym_result = sym.solve(constraints, *variables, dict=True)

        num_solutions = len(sym_result)

        if num_solutions < 1:
            logger.info("No solutions found.")
            return {}

        elif num_solutions > 1:
            raise ValueError(
                f"Found {num_solutions} solutions. System is likely underconstrained"
            )

        result = {}
        for symbol, value in sym_result[0].items():
            result[symbol.name] = value

        return result

    def resolve(
        self,
        *constraints: Real | str | sym.Expr,
        inplace: bool = True,
        sort: bool | Callable = True,
        reset_zero: Literal["pos", "neg", "both"] = "neg",
        **substitutions: Real | str | sym.Expr,
    ) -> list[tuple[Location, Operation]]:
        lw_pairs = []

        new_constraints = qwip.converter.structure(constraints, set[sym.Expr])
        new_constraints.update(
            {
                sym.Symbol(symbol) - qwip.converter.structure(value, sym.Expr)
                for symbol, value in substitutions.items()
            }
        )

        solved = self.solve_constraints(*new_constraints)
        var_set = set(solved.keys())

        tmin = tmax = None

        for loc, op in self:
            if {s.name for s in loc.free_symbols} & var_set:
                loc = loc.subs(solved)

            if op.variables() & var_set:
                op = op.resolve(**solved)

            lw_pairs.append((loc, op))

            width = 0 if op.width is sym.oo else op.width
            tmin = loc if tmin is None else min(tmin, loc)
            tmax = loc + width if tmax is None else max(tmax, loc + width)

        should_reset = (
            (reset_zero == "neg" and tmin < 0)
            or (reset_zero == "pos" and tmin > 0)
            or (reset_zero == "both")
        )

        if should_reset:
            shift = -tmin  # Need to do this bc generator is evaluated after tmin = 0
            lw_pairs = ((loc + shift, op) for loc, op in lw_pairs)
            tmax = tmax - tmin
            tmin = tmin - tmin

        if sort:
            sort_key = sort if isinstance(sort, Callable) else _default_sort_key
            lw_pairs = sorted(lw_pairs, key=sort_key)

        if inplace:
            self.lw_pairs[:] = lw_pairs
            self.width = tmax

        if not isinstance(lw_pairs, list):
            lw_pairs = list(lw_pairs)

        return lw_pairs

    @deprecated(
        version="24.6.0", removed="24.8.0", message="Use `Timeline.resolve` instead."
    )
    def resolve_waveforms(
        self, **pulse_vars: float | int
    ) -> dict[Operation, Operation]:
        """Resolves all waveform variables into concrete values.

        Args:
            pulse_vars: A mapping of string variables to variables

        Returns:
            A mapping of waveforms of original waveforms to resolved waveforms.
        """

        waveform_dict = {}

        for waves in self.locations.values():
            for idx, wave in enumerate(waves):
                new = wave.resolve(**pulse_vars)

                if new == wave:
                    continue

                waveform_dict[wave] = new
                waves[idx] = new

                self.add_constraints(**pulse_vars)

        if self.width:
            self.width = self.width.resolve(**pulse_vars)

        return waveform_dict

    @deprecated(
        version="24.6.0", removed="24.8.0", message="Use `Timeline.resolve` instead."
    )
    def resolve_locations(
        self,
        sort: bool = True,
        reset_zero: str | None = "neg",
        end_marker: str | None = "end",
        markers: dict | None = None,
        **kwargs: Location,
    ) -> dict[Location, list[Operation]]:
        """Resolves all locations into concrete times.

        Optionally time orders the location mapping and sets the earliest location
        to t = 0.

        Args:
            sort: Whether or not to time order the location mapping.
            reset_zero: Whether or not to translate the location mapping such that
                the earliest location is t = 0. Can be 'neg', 'pos', 'both', or None.
            **kwargs: Additional constraints to add to the pulse timelines
                before solving for the locations.

        Returns:
            A dictionary mapping concrete locations to lists of Operations
        """
        constraints = self.solve_constraints(**kwargs)

        locations = dict()

        t_max = Location()

        for loc, waves in self.locations.items():
            loc = loc.resolve(**constraints)
            locations[loc] = locations.get(loc, list())

            locations[loc].extend(waves)

            widths = [
                w.width.resolve(**constraints)
                for w in waves
                if not isinstance(w, InfiniteWaveform)
            ]
            t = loc + (max(widths) if widths else 0)

            t_max = t if t > t_max else t_max

        if end_marker:
            locations[t_max] = locations.get(t_max, [])
            locations[t_max] += [Marker(name=end_marker)]

            if markers is not None:
                markers[end_marker] = t_max

        if sort:
            locations = dict(sorted(locations.items(), key=lambda l: l[0]))

        t0 = next(iter(locations)) if sort else min(locations)
        should_reset = (
            (reset_zero == "neg" and t0 < Location())
            or (reset_zero == "pos" and t0 > Location())
            or (reset_zero == "both")
        )
        if should_reset:
            locations = {loc - t0: w for loc, w in locations.items()}

            if markers is not None:
                for k in markers:
                    markers[k] = markers[k] - t0

        return locations

    def translate(
        self,
        dt: LocationLike,
    ) -> Self:
        """Shifts a pulse timeline in time.

        This function translates all the waveforms in the pulse timeline by dt, which
        is equivalent to taking s(t) -> s(t - dt).

        Args:
            dt: Amount of time to translate locations by.

        Returns:
            The pulse timeline.
        """

        self.lw_pairs[:] = ((loc + dt, op) for loc, op in self)

        return self

    def transform_waveforms(
        self, transformer: Callable[[Location, Waveform], Waveform]
    ) -> int:
        """Applies a waveform transformer to every waveform in the pulse timeline.

        Args:
            transformer: A callable that takes a location, waveform pair and returns
                a possibly modified waveform.

        Returns:
            The number of waveforms that were modified by the transformer.
        """

        modified = 0

        for idx, (loc, op) in enumerate(self):
            new_op = transformer(loc, op)

            if new_op != op:
                self.lw_pairs[idx] = (loc, new_op)
                modified += 1

        return modified

    def assign_channels(self, **channels: str) -> Self:
        """Reassigns channels for all operations in the timeline.

        This method modifies the timeline in place.

        Args:
            **channels: A mapping of old channel names to new channel names.

        Returns:
            The modified timeline.
        """

        if not set(channels) & self.channels:
            return self

        for idx, (loc, op) in enumerate(self):
            if op.channel in channels:
                self.lw_pairs[idx] = (loc, op.assign_channel(channels[op.channel]))

        new_channels = [channels.get(ch, ch) for ch in self.channels]
        self.channels.clear()
        self.channels.update((ch for ch in new_channels if ch))

        return self

    def copy(self, deep: bool = True) -> Self:
        """Copies a pulse timeline.

        This function makes a copy of the pulse timeline. Defaults to making a deep copy
        but can also make a shallow copy where the constraint dict and waveform mappings are
        shared.

        Args:
            deep: Whether to make a deep copy or shallow copy. Defaults to True.

        Returns:
            The new pulse timeline.
        """

        return deepcopy(self) if deep else copy(self)

    @staticmethod
    def locations_to_channel_map(
        locations: list[tuple[sym.Expr, Operation]], *channels: str
    ) -> dict[str, list[tuple[sym.Expr, Waveform]]]:
        """Splits a location dict by channel.

        Makes a single pass through the location dict. This static method
        is provided as a convenience for processing arbitrary location maps.

        Args:
            channels: The channels to include in the channel map.

        Returns:
            A dictionary mapping channels to (location, waveform) pairs.
        """

        channel_map = {c: [] for c in channels}

        for loc, wave in locations:
            if wave.channel in channel_map:
                channel_map[wave.channel].append((loc, wave))

        return channel_map

    def get_channel_map(
        self, *channels: str
    ) -> dict[str, list[tuple[Location, Waveform]]]:
        """Splits the location dict by channel.

        Makes a single pass through the location dict.

        Args:
            channels: The channels to include in the channel map.

        Returns:
            A dictionary mapping channels to (location, waveform) pairs.
        """
        if not channels:
            channels = self.channels

        return type(self).locations_to_channel_map(self.lw_pairs, *channels)

    def plot(
        self,
        channels: list[tuple[str, ...]] | None = None,
        constraints: dict[str, Location] = {},
        filter_func: Callable[[Location, Waveform], bool] = None,
        axes: Collection[Axes] = None,
        fig_props: dict = {},
        pulse_vars: dict = {},
    ) -> Figure:
        """Plots the pulse timeline.

        This function uses a default instance of TimelinePlotter to
        render the pulse timeline. Since pulse timelines are not yet compiled
        an abstract rendering of the pulse timeline is created.

        See TimelinePlotter to customize how pulse timelines are
        rendered.

        Args:
            channels: An optional list of channel groups. Groups can be specified as a
                tuple of channel names.
            constraints: A constraint dict to apply to the pulse timeline
                before resolving locations.
            filter_func: A callable used to filter the waveforms. Takes a
                location and waveform and returns True if the waveform should be
                included.
            axes: A set of axes on which to plot the pulse timeline. Can be
                used to plot the pulse timeline on an existing figure. If
                None, a new figure is created.
            fig_props: Optional arguments passed to TimelinePlotter.make_axes

        Returns:
            The matplotlib figure containing the plot axes.
        """
        return TimelinePlotter().plot(
            self,
            channels,
            constraints,
            filter_func,
            axes,
            fig_props,
            pulse_vars,
        )

    @deprecated(
        version="24.6.0",
        removed="24.8.0",
        message="Use the timeline as an iterable directly instead.",
    )
    def get_location_pairs(self) -> list[tuple[Location, Waveform]]:
        """Returns a list of all `(loc, wave)` pairs in the location mapping."""

        return list(self)

    def __getitem__(self, key: LocationLike) -> list[Waveform]:
        key = qwip.converter.structure(key, sym.Expr)

        return [op for loc, op in self if key == loc]

    def __contains__(self, waveform: Waveform) -> bool:
        """Checks if the waveform exists in the pulse timeline."""

        return waveform in self.operations

    def __iter__(self) -> Iterator[sym.Expr, Operation]:
        """Iterate over the location mapping."""
        yield from self.lw_pairs

    def __add__(self, other: Self | Operation) -> Self:
        """Adds two pulse timelines.

        The sum of two pulse timelines s(t) and r(t) is equivalent to the
        pointwise addition at every point in time.

        Args:
            other: The other pulse timeline to add.

        Returns:
            A new pulse timeline equal to s(t) + r(t).
        """

        tmln = Timeline()

        match other:
            case Timeline():
                lw_pairs = other.lw_pairs
                channels = other.channels
                constraints = other.constraints
            case Operation():
                lw_pairs = [(0, other)]
                channels = {other.channel} if other.channel else set()
                constraints = []
            case np.ndarray():  # Send to Sequence __radd__
                return NotImplemented
            case _:
                raise TypeError(
                    f"Can only add Timeline and Operation to Timeline, not "
                    f"'{type(other).__name__}'"
                )

        tmln.lw_pairs[:] = self.lw_pairs + lw_pairs
        tmln.channels.update(self.channels | channels)
        tmln.constraints.update(self.constraints, constraints)

        return tmln

    def __radd__(self, other: Self | Operation) -> Self:
        return self.__add__(other)


structure_new_timeline = make_attrs_structure_fn(Timeline)


def structure_timeline(obj, cls):
    if (
        isinstance(obj, dict)
        and "locations" in obj
        or isinstance(obj.get("constraints"), dict)
    ):
        lw_pairs = [
            (loc, qwip.converter.structure(w, Waveform))
            for loc, waves in obj.get("locations", {}).items()
            for w in waves
        ]

        constraints = set()
        for name, expr in obj.get("constraints", {}).items():
            constraints.add(sym.Symbol(name) - qwip.converter.structure(expr, sym.Expr))

        width = obj.get("width", None)

        return Timeline(lw_pairs=lw_pairs, constraints=constraints, width=width)

    return structure_new_timeline(obj, cls)


qwip.converter.register_structure_hook(Timeline, structure_timeline)


@qdefine
class TimelinePlotter:
    """Plotter for pulse timelines."""

    axsize: tuple[float, float] = (8, 1)
    sort_channels: bool = True
    separate_none: bool = False
    channel_grouper: Callable[[Self, Collection[str]], list[tuple[str, ...]]] | None = (
        None
    )

    def make_axes(
        self,
        n: int,
        axsize: tuple[float, float] | None = None,
        sharex: bool = True,
        sharey: bool = True,
        **props,
    ) -> Figure:
        """Creates a matplotlib figure and axes.

        Args:
            n: Number of axes.
            axsize: The size (width, height) in inc
        """
        if "figsize" not in props:
            axsize = axsize or self.axsize
            width, height = axsize

            if width == height == ...:
                width, height = (8, 1)
            elif width is ...:
                width = 8 / height
            elif height is ...:
                height = 1 / 8 * width

            props["figsize"] = (width, n * height)

        fig, _ = plt.subplots(n, 1, sharex=sharex, sharey=sharey, **props)

        return fig

    def group_channels(
        self, channels: list[tuple[str, ...]] | None, channel_map: TChannelMap
    ) -> list[tuple[str, ...]]:
        # TODO: this can probably be removed since Channels are now just strings.
        if channels:
            return channels

        if self.channel_grouper:
            return self.channel_grouper(self, channel_map.keys())

        return self.default_channel_grouper(channel_map.keys())

    def default_channel_grouper(self, channels):
        if self.sort_channels:

            def get_name(maybe_channel):
                if maybe_channel:
                    return maybe_channel.name
                return ""

            channels = sorted(channels, key=get_name)

        if self.separate_none:
            channels = [(ch,) for ch in channels]
        else:
            channels = [(ch, None) for ch in channels if ch]

        return channels

    def plot_panel(
        self,
        ax: Axes,
        channel_map: TChannelMap,
        filter_func: Callable[[Location, Waveform], bool] = None,
        pulses: dict[Waveform, int] | None = None,
        pulse_vars: dict = {},
    ) -> None:
        seen = set()
        for loc_waves in channel_map.values():
            for loc, wave in loc_waves:
                # Avoids repeating the exact same waveform
                if (loc, wave) in seen:
                    continue

                seen.add((loc, wave))

                if filter_func and not filter_func(loc, wave):
                    continue

                label = None
                if pulses is not None and wave not in pulses:
                    pulses[wave] = len(pulses)
                    label = wave.name

                props = dict(
                    color=f"C{pulses[wave]}",
                    label=label,
                    alpha=0.5,
                )

                wave = wave.resolve(**pulse_vars)
                self.add_waveform_to_axes(wave, loc, ax, **props)

        ax.set_ylabel("\n".join(ch.name for ch in channel_map if ch))

    def plot(
        self,
        tmln: Timeline,
        channels: list[tuple[str, ...]] | None = None,
        constraints: dict[str, Location] = {},
        filter_func: Callable[[Location, Waveform], bool] = None,
        axes: Collection[Axes] = None,
        fig_props: dict = {},
        pulse_vars: dict = {},
    ) -> Figure:
        locations = tmln.resolve_locations(**constraints)
        channel_map = Timeline.locations_to_channel_map(locations, *tmln.channels, None)

        channels = self.group_channels(channels, channel_map)

        if axes is None:
            fig = self.make_axes(len(channels), **fig_props)
            axes = fig.axes

            if isinstance(axes, Axes):
                axes = np.array([axes])

        pulses = {}

        for ax_id, chan_group in enumerate(channels):
            self.plot_panel(
                axes[ax_id],
                {ch: channel_map[ch] for ch in chan_group},
                filter_func=filter_func,
                pulses=pulses,
                pulse_vars=pulse_vars,
            )

        figwidth, _ = fig.get_size_inches()

        h, l = all_legend_handles_labels(axes)
        axes[0].legend(
            h,
            l,
            mode="expand",
            bbox_to_anchor=(0, 1.05, 1, 0.05),
            loc="lower left",
            ncols=min(figwidth // 2, len(l)),
            borderaxespad=0,
        )

        axes[0].set_ylim(0, 1)
        axes[-1].set_xlabel("Time (s)")

        return fig

    @singledispatchmethod
    def add_waveform_to_axes(
        self, wave: Waveform, loc: Location, ax: Axes, **props
    ) -> None:
        start, end = loc.offset, loc.offset + wave.width.offset
        wfunc = CosineRampWaveform(amplitude=wave.amplitude, width=(end - start))

        ts = np.linspace(start, end)
        ax.fill_between(ts, y1=wfunc(ts, t0=start), **props)

    @add_waveform_to_axes.register(Marker)
    def _(self, wave: Waveform, loc: Location, ax: Axes, **props) -> None:
        start = loc.offset
        ax.axvline(start, **props)


__all__ = ["Timeline", "TimelinePlotter"]
