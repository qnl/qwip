import itertools as it
import re
from collections.abc import Sequence as TSequence
from typing import Any, Self

import numpy as np
import pandas as pd
from loguru import logger
from numpy.typing import NDArray

import qwip
from qwip._cattr import make_attrs_unstructure_fn
from qwip.attrs import qdefine
from qwip.sequencer.timeline import Timeline

DIM_REGEX = re.compile(r"d(?P<axis>\d+)")

SEQUENCE_FUNCTIONS = {}


def sequence_implements(np_function):
    def decorator(func):
        SEQUENCE_FUNCTIONS[np_function] = func
        return func

    return decorator


@qdefine(init=False, slots=False, repr=False, eq=False, order=False)
class Sequence(np.ndarray):
    labels: tuple[pd.Index | None, ...]

    def __new__(
        cls,
        array: NDArray[Timeline],
        /,
        **labels,
    ):
        # Turn array into ndarray and return view as Sequence
        # If array is already a subclass of ndarray, it will pass through
        # np.asanyarray unchanged.
        obj = np.asanyarray(array, dtype=object).view(cls)

        if len(labels) > obj.ndim:
            raise ValueError(
                f"Number of labels exceeds number of sequence dimensions {obj.shape}."
            )

        validated = []
        names = set()
        for name, label in labels.items():
            axis = len(validated)
            match label:
                case pd.MultiIndex():
                    label.name = "_".join(label.names)
                case pd.Index(name=label_name):
                    if label_name is None:
                        label.name = name
                    elif name != label_name:
                        logger.warning(
                            f"Index name '{label_name}' does not match keyword argument "
                            f"'{name}'. Using '{label_name}' for axis {axis}"
                        )
                case None:
                    label = pd.RangeIndex(0, obj.shape[axis], name=name)
                case _:
                    label = pd.Index(label, name=name)

            if label.shape[0] != obj.shape[axis]:
                raise ValueError(
                    f"Label '{name}' has length {label.shape[0]} that does not match "
                    f"sequence dimension {obj.shape[axis]} for axis {axis}."
                )

            if name in names:
                raise ValueError(
                    f"Sequence axis names must be unique, got duplicate name '{name}' "
                    f"for axis {axis}."
                )

            names.add(label.name)
            validated.append(label)

        for axis in range(len(validated), obj.ndim):
            # Use existing or default label if not explicitly specified.
            label = obj.labels[axis]
            if label.name in names:
                raise ValueError(
                    f"Reserved axis name '{label.name}' is used on the wron axis."
                )
            validated.append(label)

        obj.labels = tuple(validated)
        obj.__array_finalize__()

        return obj

    def __array_finalize__(self, obj: NDArray[Timeline] | None = None, /) -> None:
        # No additional cleanup necessary if this is explicit construction
        if obj is None:
            return

        if not hasattr(self, "labels"):
            # We copy names from obj if it exists and obj matches the correct shape
            if self.shape == obj.shape and hasattr(obj, "labels"):
                self.labels = obj.labels
            # otherwise set to default
            else:
                self.labels = tuple(
                    pd.RangeIndex(0, d, name=f"d{axis}")
                    for axis, d in enumerate(self.shape)
                )

    def __repr__(self) -> str:
        names = f"names={self.names}" + ", "
        prefix = "    "
        arr = prefix + np.array2string(self, prefix=prefix)
        return f"Sequence({names}shape={self.shape}\n{arr}\n)"

    def _expand_basic_index(self, index: tuple) -> tuple:
        """Expands out ellipses in numpy basic indices.

        This expands the index according to the shape of self. According to
        the numpy [documentation](https://numpy.org/doc/stable/user/basics.indexing.html#dimensional-indexing-tools)
        ... will expand out to as many slice(None) as needed to match the
        shape of the ndarray.

        Args:
            index: The numpy index to expand that may include ...

        Returns:
            An equivalent index with every dimension expanded out.
        """

        if not isinstance(index, tuple):
            index = (index,) if index is ... else (index, ...)
        elif ... not in index:
            return index

        # Number of slices to create
        n = len(self.shape) - len(tuple(i for i in index if i not in (..., np.newaxis)))

        # Must return a single element interable if not ellipse for itertools.chain
        def replace_ellipsis(idx: tuple):
            return (slice(None) for _ in range(n)) if idx is ... else (idx,)

        return tuple(it.chain(*(replace_ellipsis(i) for i in index)))

    def _get_labels_from_index(self, index: tuple[int, ...]) -> tuple[pd.Index, ...]:
        """Determines new labels for a view of self given the index.

        Args:
            index: The expanded index with no ellipses.

        Returns:
            A list of updated labels.
        """
        labels = []

        old_axis = 0
        for new_axis, idx in enumerate(index):
            match idx:
                case slice():
                    labels.append(idx := self.labels[old_axis][idx])
                    old_axis += 1
                # A new axis is being created here with dimension 1
                case None:
                    labels.append(pd.RangeIndex(1, name=f"d{new_axis}"))
                case int():
                    old_axis += 1

        for axis, label in enumerate(labels):
            default = f"d{axis}"
            if isinstance(label, pd.MultiIndex):
                label.name = "_".join(label.names)

            if DIM_REGEX.fullmatch(label.name) and label.name != default:
                label.name = default

        return tuple(labels)

    def __getitem__(self, key):
        # Shortcut for getting a label
        if isinstance(key, str):
            for label in self.labels:
                if label.name == key:
                    return label
            else:
                raise KeyError(f"'{key}'")

        if _is_advanced_index(key):
            return NotImplemented

        obj = super().__getitem__(key)

        # If indexing leads to a single valued timeline, e.g. seq[0] for a 1D sequence
        if not isinstance(obj, type(self)):
            return obj

        expanded = self._expand_basic_index(key)
        logger.trace(f"Expanded form of {key} is {expanded}")
        labels = self._get_labels_from_index(expanded)  # Set names
        obj.labels = labels

        return obj

    def __array_ufunc__(
        self, ufunc: np.ufunc, method: str, *inputs, out=None, **kwargs
    ):
        outputs = out if out else (None,) * ufunc.nout

        args = (
            arr.view(np.ndarray) if isinstance(arr, Sequence) else arr for arr in inputs
        )

        results = getattr(ufunc, method)(*args, **kwargs)
        labels = broadcast_labels(*(seq for seq in inputs if isinstance(seq, Sequence)))

        if results is NotImplemented:
            return NotImplemented

        if method == "at":
            return

        if ufunc.nout == 1:
            results = (results,)

        results = tuple(
            (np.asanyarray(result).view(type(self)) if output is None else output)
            for result, output in zip(results, outputs)
        )

        if len(results) == 1:
            results = results[0]

            if method == "reduce":
                axis = kwargs.get("axis", 0)

                if axis is None:
                    labels = tuple()
                else:
                    axis = (axis,) if isinstance(axis, int) else axis
                    axis = tuple(d + inputs[0].ndim if d < 0 else d for d in axis)

                    if kwargs.get("keepdims", False):
                        labels = tuple(
                            pd.RangeIndex(1, name=lb.name) if i in axis else lb
                            for i, lb in enumerate(labels)
                        )
                    else:
                        labels = tuple(
                            lb for i, lb in enumerate(labels) if i not in axis
                        )

            for axis, label in enumerate(results.labels):
                if len(label) != results.shape[axis]:
                    labels[axis] = pd.RangeIndex(results.shape[axis], name=label.name)

            results.labels = tuple(labels)

        return results

    def __array_function__(self, func, types, args, kwargs):
        if func not in SEQUENCE_FUNCTIONS:
            return NotImplemented

        if not all(issubclass(t, type(self)) for t in types):
            return NotImplemented

        return SEQUENCE_FUNCTIONS[func](*args, **kwargs)

    @property
    def names(self) -> tuple[str]:
        return tuple(lb.name for lb in self.labels)

    @property
    def T(self) -> Self:
        """Returns the transpose of a sequence.

        This is necessary to ensure that seq.T.names has the correct
        ordering of axis names.

        Returns:
            A transposed view of the of the sequence.
        """
        return self.transpose()

    def transpose(self, *axes: int) -> Self:
        """Reverses or permutes axis of the sequence.

        Args:
            axes: Specifies the permutation of the axes. If None, the axes are
                reversed. Can be a tuple of ints or n ints.

        Returns:
            Returns a view of the sequence with the permuted axes.
        """
        seq = super().transpose(*axes)

        if len(axes) == 1:
            axes = axes[0]

        if axes is None or not len(axes):
            seq.labels = tuple(reversed(self.labels))
            return seq

        seq.labels = tuple(self.labels[i] for i in axes)

        return seq

    @classmethod
    def empty(
        cls,
        shape: tuple[int, ...],
        names: tuple[str, ...] | None = None,
        **labels: np.ndarray,
    ) -> Self:
        """Creates a Sequence of the specified shape with empty Timelines.

        Args:
            shape: The desired shape of the output sequence.
            names: Names to attach to the axis dimensions.
            labels: Labels to attach to the axis dimensions.
        """

        arr = np.empty(shape, dtype=object)

        for index in np.ndindex(*arr.shape):
            arr[index] = Timeline()

        if names is not None:
            labels = {}

        return cls(arr, **labels)

    @classmethod
    def sweep(cls, tmln: Timeline, /, name=None, label=None, **params) -> Self:
        """Create a sequence from the sequence element."""

        shape = min(len(arr) for arr in params.values())
        name = name or "_".join(params)

        values = list(zip(*params.values()))

        if label is None:
            if len(params) == 1:
                label = params[name]
            else:
                label = np.empty(shape, dtype=object)
                label[:] = values
        elif label.shape[0] != len(values):
            raise ValueError(
                f"Provided label must have length {len(values)} that matches "
                f"sequence shape."
            )

        seq = Sequence.empty((shape,), names=(name,), **{name: label})

        for i, vals in enumerate(values):
            new = tmln.copy()

            update = {n: v for n, v in zip(params, vals)}
            new.add_constraints(**update)
            new.resolve_waveforms(**update)

            seq[i] = new

        return seq

    @classmethod
    def product(cls, tmln: Timeline, /, **params) -> Self:
        """Create a sequence from the sequence element."""
        shape = tuple(len(arrs) for arrs in params.values())
        names = tuple(params)

        seq = Sequence.empty(shape, names=names, **params)

        for i, vals in enumerate(it.product(*params.values())):
            new = tmln.copy()

            update = {n: v for n, v in zip(names, vals)}
            new.add_constraints(**update)
            new.resolve_waveforms(**update)

            seq.flat[i] = new

        return seq


def _expand_names(names: tuple, shape: tuple[int]) -> tuple:
    """Expands out ellipses in names to match shape.

    This function expands out any tuples of names containing ... to match
    the number of dimensions specified by shape. Replace ... with as many
    None values as necessary. The name tuple is also validated to ensure
    that the total length matches the number of dimensions (unless ...
    is included) and that no duplicate names exist (except for None).

    Args:
        names: The tuple of names passed to the constructor.
        shape: The shape of the array that the names will be attached to.

    Returns:
        An equivalent tuple of names with every dimension expanded out.

    Raises:
        IndexError: If names contains more than one ellipsis.
    """

    if ... not in names:
        return names

    if names.count(...) > 1:
        raise IndexError("names can only contain a single ellipsis ('...')")

    num_dims = len(shape)
    num_to_expand = num_dims - len(names) + 1

    def replace_ellipsis(n):
        return (None for _ in range(num_to_expand)) if n is ... else (n,)

    return tuple(it.chain(*(replace_ellipsis(n) for n in names)))


def _is_advanced_index(index: TSequence) -> True:
    """Determines if an index triggers numpy advanced indexing.

    See the numpy [documentation](https://numpy.org/doc/stable/user/basics.indexing.html)
    for more details on indexing.

    Args:
        index: An index passed to `ndarray.__getitem__`.

    Returns:
        True if the index would trigger advanced indexing under numpy rules.
    """

    # Check if index is a non-tuple sequence object

    int_or_bool = "biu"

    def is_sequence_or_np_index(i):
        return isinstance(i, TSequence) or (
            isinstance(i, np.ndarray) and i.dtype.kind in int_or_bool
        )

    return (is_sequence_or_np_index(index) and not isinstance(index, tuple)) or (
        isinstance(index, tuple) and any(is_sequence_or_np_index(i) for i in index)
    )


def _set_labels(
    obj: Sequence, labels: dict[str, TSequence], should_raise: bool = False
) -> Sequence:
    """Sets labels on a Sequence.

    This function is used in both the explicit constructor and
    `__array_finalize__` to copy labels from a source dict to the Sequence
    object being created. Label arrays are copied by reference when the
    Sequence object is a view of another ndarray or Sequence. Labels are
    automatically converted to ndarrays.

    Args:
        obj: The Sequence object to attach the labels to.
        labels: The source dictionary from which labels should be copied.
        should_raise: Whether or not to raise an exception or silently pass.

    Returns:
        The Sequence object.
    """
    for n, arr in labels.items():
        try:
            idx = obj.names.index(n)

        except ValueError as e:
            if should_raise:
                raise ValueError(
                    f"'{n}' is not an axis name. names = {obj.names}"
                ) from e

            continue

        if len(arr) != obj.shape[idx]:
            if should_raise:
                raise ValueError(
                    f"{arr} has shape {arr.shape} which does not match shape "
                    f"{obj.shape} for dimension {idx}."
                )

            continue

        obj.labels[n] = np.asarray(arr)

    return obj


def _is_default_label(label: pd.Index, dim: int) -> bool:
    """Determines whether the given axis has a default label.

    A default label is defined as matching an autogenerated label, which has a name of
    the form `r"d(\\d+)"` and values `[0, 1, 2, ..., N-1]` for an axis of dimension `N`.

    Args:
        label: The sequence label to check.
        axis: The axis level of the sequence to check.

    Returns:
        `True` if the axis label for the sequence matches a "default" format.
    """
    return bool(DIM_REGEX.fullmatch(label.name)) and bool(
        label.equals(pd.RangeIndex(0, dim))
    )


def _combine_indices(*indices, dim: int) -> pd.Index:
    """Combines a set of pandas indices into a single index.

    If only a single pandas index is given, it will be returned as is. Otherwise they
    will be combined into a single `pd.MultiIndex` instance where the levels match the
    order in which the indices are given. If any index has length 1, it will be
    broadcasted to match the same length of the other indices. If any indices are
    duplicates, only the first will be kept.

    Args:
        *indices: The set of indices to combine.
        dim: The length of the final index. All indices should either have length equal
            to 1 or `dim`.

    Returns:
        The resulting combined index.
    """
    match len(indices):
        case 0:
            raise ValueError("No indices to combine!")
        case 1 if len(indices) == dim:
            return indices[0]

    index_values = []
    names = []

    for idx in indices:
        if len(idx.values) == 1:
            values = np.repeat(idx.values, dim)
        else:
            values = idx.values

        if isinstance(idx, pd.MultiIndex):
            index_values += [*zip(*values)]
            names = names + idx.names
        else:
            index_values.append(values)
            names = names + [idx.name]

    combined = pd.DataFrame(dict(zip(names, index_values)))
    # set_index is necessary here b/c drop_duplicates ignores indices, which after
    # transposing corresponds to the index names. This ensures that two indices with
    # different names but the same values will both be kept.
    combined = combined.T.reset_index().drop_duplicates().set_index("index").T

    if len(combined.columns) == 1:
        return pd.Index(combined[combined.columns[0]])

    label = pd.MultiIndex.from_frame(combined)
    label.name = "_".join(label.names)
    return label


def broadcast_labels(*seqs) -> list[pd.Index]:
    """Returns the resulting labels from broadcasting one or more sequences.

    Explicit sequence labels take precedence over default sequence labels. If multiple
    sequences have explicit labels along the same axis, a combined multiiindex label
    will be created for that axis with index values from each sequence.

    Args:
        *seqs: The sequences to broadcast.

    Returns:
        The labels for the resulting broadcasted sequence.
    """
    out_shape = np.broadcast_shapes(*(s.shape for s in seqs))
    ndim = len(out_shape)

    labels = []
    for axis in range(ndim):
        axis_labels = []
        for seq in seqs:
            # Broadcasted dimension
            if ndim - axis > seq.ndim:
                continue

            # Broadcasted array shapes are right aligned
            seq_axis = seq.ndim - (ndim - axis)

            # Ignore default axis labels
            if _is_default_label(seq.labels[seq_axis], seq.shape[seq_axis]):
                continue

            axis_labels.append(seq.labels[seq_axis])

        if axis_labels:
            label = _combine_indices(*axis_labels, dim=out_shape[axis])
            labels.append(label)
        # If all sequences have default axis labels then use default.
        else:
            labels.append(pd.RangeIndex(out_shape[axis], name=f"d{axis}"))

    return labels


@sequence_implements(np.array2string)
def array2string(a, **kwargs):
    return np.array2string(np.asarray(a), **kwargs)


@sequence_implements(np.concatenate)
def concatenate(
    sequences: TSequence[Sequence], axis: int = 0, **kwargs: Any
) -> Sequence:
    """Concatenates sequences along the specified axis.

    Args:
        sequences: An iterable of sequences to concatenate.
        axis: The axis along which to concatenate the sequences.

    Returns:
        The concatenated sequences.
    """
    arr_views = tuple(np.asarray(arr) for arr in sequences)
    seq = np.concatenate(arr_views, axis=axis, **kwargs).view(Sequence)

    if len(sequences) == 1:
        seq.labels = (idx.copy() for idx in sequences[0].labels)

    if axis < 0:
        axis = seq.ndim + axis

    labels = []
    for a in range(seq.ndim):
        axis_labels = [s.labels[a] for s in sequences]
        dims = [s.shape[a] for s in sequences]
        if a == axis:
            if all([_is_default_label(lb, dim) for lb, dim in zip(axis_labels, dims)]):
                label = pd.RangeIndex(seq.shape[a], name=f"d{a}")
            else:
                label = axis_labels[0].append(axis_labels[1:])
                label.name = label.name or f"d{axis}"
        else:
            axis_labels = [
                lb for lb in axis_labels if not _is_default_label(lb, dims[a])
            ]

            if axis_labels:
                label = _combine_indices(*axis_labels, dim=dims[a])
            else:
                label = pd.RangeIndex(seq.shape[a], name=f"d{a}")

        labels.append(label)

    seq.labels = tuple(labels)

    return seq


@sequence_implements(np.stack)
def stack(
    seqs: Sequence,
    axis: int = 0,
    label: pd.Index | str | None = None,
    **kwargs,
) -> Sequence:
    """Joins sequences along a new axis.

    Args:
        seqs: An iterable of sequences to concatenate.
        axis: Specifies the new axis in the stacked sequences.
        label: Labels for the new axis. Must match the number of sequences. To specify
            a label name but not label values, a string can be given.

    Returns:
        The joined sequences.
    """
    arr_views = tuple(np.asarray(arr) for arr in seqs)
    seq = np.stack(arr_views, axis=axis, **kwargs).view(Sequence)

    if axis < 0:
        axis = seq.ndim + axis

    labels = broadcast_labels(*seqs)
    names = [lb.name for lb in labels]

    N = len(seqs)
    default_name = f"d{axis}"
    match label:
        case None:
            label = pd.RangeIndex(N, name=default_name)
        case str():
            label = pd.RangeIndex(N, name=label)
        case pd.Index():
            label.name = label.name or default_name
        case _:
            raise ValueError(f"Label must be a pd.Index, str, or `None`, got {label}.")

    if name := label.name in names:
        logger.warning(
            f"Axis name {name} is already in names. Falling back to default name "
            f"'d{axis}' for axis {axis}."
        )

        label.name = f"d{axis}"

    if len(label) != N:
        raise ValueError(
            f"Label length ({len(label)}) does not match number of sequences ({N})."
        )

    labels.insert(axis, label)

    for a, label in enumerate(labels):
        if DIM_REGEX.fullmatch(label.name):
            label.name = f"d{a}"

    seq.labels = tuple(labels)
    return seq


@sequence_implements(np.reshape)
def reshape(seq: Sequence, shape: tuple[int, ...], **kwargs: Any) -> Self:
    """Reshapes the sequence.

    Reshaping a sequence creates a view of the sequence but does not
    copy over the axis names and labels in most cases, since this is
    not a sensible thing to do most of the time.

    Args:
        seq: The sequence to reshape.
        shape: The new shape of the sequence.
        **kwargs: Keyword arguments are passed to numpy reshape.

    Returns:
        A view of the sequence with the specified shape.
    """
    return seq.reshape(shape, **kwargs)


@sequence_implements(np.transpose)
def transpose(
    seq: Sequence,
    axes: tuple[int, ...] | None = None,
) -> Sequence:
    """Reverses or permutes axis of a sequence.

    Args:
        seq: The sequence to transpose.
        axes: Specifies the permutation of the axes. If None, the axes are
            reversed.

    Returns:
        The transposed sequence. A view is returned if possible.
    """
    return seq.transpose(axes)


@sequence_implements(np.add)
def add(seq1: Sequence, seq2: Sequence, /, **kwargs):
    return seq1.add(seq2, **kwargs)


@sequence_implements(np.sum)
def sum(seq: Sequence, axis: tuple[int, ...] | int | None = None, **kwargs) -> Sequence:
    """Sum of sequence elements over a specified axis.

    Args:
        seq: The sequence to sum.
        axis: The axis or axes to add. If axis=None, the entire sequence
            is summed.

    Returns:
        The resulting sequence.
    """
    return seq.sum(axis, **kwargs)


def make_sequence_structure_fn(cls):
    def structure_fn(val, cls):
        if isinstance(val, cls):
            return val

        shape = val["shape"]

        tp = Timeline
        for _ in shape:
            tp = list[tp]

        data = qwip.converter.structure(val["data"], tp)

        return Sequence(data, **val["labels"])

    return structure_fn


def make_sequence_unstructure_fn(cls):
    def unstructure_fn(obj):
        return {
            "names": obj.names,
            "labels": {
                label.name: qwip.converter.unstructure(label.to_numpy())
                for label in obj.labels
            },
            "shape": obj.shape,
            "data": qwip.converter.unstructure(obj.tolist()),
        }

    return unstructure_fn


qwip.converter.register_structure_hook_factory(
    lambda cls: issubclass(cls, Sequence), make_sequence_structure_fn
)

qwip.converter.register_unstructure_hook_factory(
    lambda cls: issubclass(cls, Sequence), make_sequence_unstructure_fn
)


__all__ = [
    "Sequence",
    "array2string",
    "concatenate",
    "stack",
    "reshape",
    "transpose",
    "add",
    "sum",
]
