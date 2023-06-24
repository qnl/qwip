import itertools as it
from collections.abc import Sequence as TSequence

import numpy as np
from loguru import logger
from numpy.typing import NDArray
from typing_extensions import Self

from qwip.attrs import qdefine
from qwip.sequencer.elements import SequenceElement

SEQUENCE_FUNCTIONS = {}


def sequence_implements(np_function):
    def decorator(func):
        SEQUENCE_FUNCTIONS[np_function] = func
        return func

    return decorator


@qdefine(init=False, slots=False, repr=False, eq=False, order=False)
class Sequence(np.ndarray):
    names: tuple[str | None, ...]
    labels: dict[str, np.ndarray]

    def __new__(
        cls,
        array: NDArray[SequenceElement],
        names: tuple[str, ...] | None = None,
        **labels,
    ):
        # Turn array into ndarray and return view as Sequence
        # If array is already a subclass of ndarray, it will pass through
        # np.asanyarray unchanged.
        obj = np.asanyarray(array, dtype=object).view(cls)

        if names is None and not labels:
            return obj

        # Validate names
        if names is not None:
            names = _expand_names(names, obj.shape)

            if len(names) != len(obj.shape):
                raise ValueError(
                    f"Length of axis names {names} does not match shape {obj.shape}."
                )

            unique_names = set()
            for n in names:
                if n in unique_names:
                    raise ValueError(f"{names} contains a duplicate name!")
                if n is not None:
                    unique_names.add(n)

            obj.names = names

        if labels:
            obj.labels = dict()
            _set_labels(obj, labels, should_raise=True)

        obj.__array_finalize__()

        return obj

    def __array_finalize__(
        self, obj: NDArray[SequenceElement] | None = None, /
    ) -> None:
        # No additional cleanup necessary if this is explicit construction
        if obj is None:
            return

        if not hasattr(self, "names"):
            # We copy names from obj if it exists and obj matches the correct shape
            if hasattr(obj, "names") and self.shape == obj.shape:
                self.names = obj.names
            # otherwise set to default
            else:
                self.names = (None,) * len(self.shape)

        if not hasattr(self, "labels"):
            self.labels = dict()

            if hasattr(obj, "labels"):
                _set_labels(self, obj.labels, should_raise=False)

    def __repr__(self) -> str:
        names = (
            "" if all(n is None for n in self.names) else f"names={self.names}" + ", "
        )
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

    def _get_names_from_index(self, index: tuple) -> tuple:
        """Determines new names for a view of self given the index.

        Args:
            index: The expanded index with no ellipses.

        Returns:
            A tuple of updated names
        """
        names = []

        dim = 0
        for idx in index:
            if isinstance(idx, slice):
                names.append(self.names[dim])
                dim += 1
            elif idx is None:
                names.append(None)
            else:
                dim += 1

        return tuple(names)

    def __getitem__(self, key):
        if _is_advanced_index(key):
            return NotImplemented

        obj = super().__getitem__(key)

        # If indexing leads to a single valued sequence element
        if not isinstance(obj, type(self)):
            return obj

        expanded = self._expand_basic_index(key)
        logger.debug(f"Expanded form of {key} is {expanded}")
        obj.names = self._get_names_from_index(expanded)  # Set names

        slices = tuple(idx for idx in expanded if idx is not None)
        # Copy over label views
        for level, n in enumerate(self.names):
            if n not in obj.names or n not in self.labels:
                continue

            logger.debug(
                f"Slicing {slices[level]} from label for {n} which was level {level}"
            )
            obj.labels[n] = self.labels[n][slices[level]]

        return obj

    def __array_ufunc__(
        self, ufunc: np.ufunc, method: str, *inputs, out=None, **kwargs
    ):
        outputs = out if out else (None,) * ufunc.nout

        args = (
            arr.view(np.ndarray) if isinstance(arr, Sequence) else arr for arr in inputs
        )

        results = getattr(ufunc, method)(*args, **kwargs)

        names, labels = broadcast_names_and_labels(
            *(seq for seq in inputs if isinstance(seq, Sequence)),
            raise_on_conflict=ufunc.__name__ not in ("equal", "not_equal"),
        )

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

            if method == "reduce" and not kwargs.get("keepdims", False):
                axis = kwargs.get("axis", 0)

                if axis is None:
                    names = tuple()
                else:
                    axis = (axis,) if isinstance(axis, int) else axis
                    axis = tuple(d + inputs[0].ndim if d < 0 else d for d in axis)

                    names = tuple(n for i, n in enumerate(names) if i not in axis)

            results.names = names
            for dim, n in enumerate(results.names):
                if n in labels and len(labels[n]) == results.shape[dim]:
                    results.labels[n] = labels[n]

        return results

    def __array_function__(self, func, types, args, kwargs):
        if func not in SEQUENCE_FUNCTIONS:
            return NotImplemented

        if not all(issubclass(t, type(self)) for t in types):
            return NotImplemented

        return SEQUENCE_FUNCTIONS[func](*args, **kwargs)

    @property
    def T(self):
        """Returns the transpose of a sequence.

        This is necessary to ensure that seq.T.names has the correct
        ordering of axis names.

        Returns:
            A transposed view of the of the sequence.
        """
        return self.transpose()

    def transpose(self, *axes):
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
            seq.names = tuple(reversed(self.names))
            return seq

        seq.names = tuple(self.names[i] for i in axes)

        return seq

    @classmethod
    def empty(
        cls,
        shape: tuple[int, ...],
        names: tuple[str, ...] | None = None,
        **labels: np.ndarray,
    ) -> Self:
        """Creates a Sequence of the specified shape with empty SequenceElements.

        Args:
            shape: The desired shape of the output sequence.
            names: Names to attach to the axis dimensions.
            labels: Labels to attach to the axis dimensions.
        """

        arr = np.empty(shape, dtype=object)

        for index in np.ndindex(*arr.shape):
            arr[index] = SequenceElement()

        return cls(arr, names, **labels)

    @classmethod
    def sweep(cls, se: SequenceElement, /, name=None, label=None, **params) -> Self:
        """Create a sequence from the sequence element."""

        shape = min(len(arr) for arr in params.values())
        name = name or ",".join(params)

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
            new = se.copy()

            update = {n: v for n, v in zip(params, vals)}
            new.add_constraints(**update)
            new.resolve_waveforms(**update)

            seq[i] = new

        return seq

    @classmethod
    def product(cls, se: SequenceElement, /, **params) -> Self:
        """Create a sequence from the sequence element."""
        shape = tuple(len(arrs) for arrs in params.values())
        names = tuple(params)

        seq = Sequence.empty(shape, names=names, **params)

        for i, vals in enumerate(it.product(*params.values())):
            new = se.copy()

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

    This function os used in both the explicit constructor and
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


def broadcast_names_and_labels(*seqs, raise_on_conflict=False):
    b = np.broadcast(*seqs)

    def get_name(seq, dim):
        if b.ndim - seq.ndim > dim:
            return None

        return seq.names[seq.ndim - b.ndim + dim]

    seq_names = tuple(
        tuple(get_name(seq, dim) for seq in seqs) for dim in range(b.ndim)
    )

    names = []
    labels = {}
    for dim, axis_names in enumerate(seq_names):
        unique_names = set(n for n in axis_names if n is not None)
        if len(unique_names) == 0:
            names.append(None)
            continue
        elif len(unique_names) > 1:
            if raise_on_conflict:
                raise ValueError(
                    f"All names along axis {dim} must be the same. {seq_names[dim]}"
                )

            names.append(None)
            continue

        name = next(iter(unique_names))  # Get the axis name
        names.append(name)

        axis_labels = []

        for s in seqs:
            label = s.labels.get(name)

            # append labels for each array to labels if it exists
            if label is not None:
                if len(label) != b.shape[dim]:
                    label = np.broadcast_to(label, (b.shape[dim],))
                axis_labels.append(label)

        if axis_labels:
            unique_values = np.unique(np.stack(axis_labels), axis=0)
            # Assign labels only if all labels are the same
            if unique_values.shape[0] == 1:
                labels[name] = axis_labels[0]

    return names, labels


@sequence_implements(np.array2string)
def array2string(a, **kwargs):
    return np.array2string(np.asarray(a), **kwargs)


@sequence_implements(np.concatenate)
def concatenate(sequences: TSequence[Sequence], axis=0, **kwargs):
    """Concatenates sequences along the specified axis.

    Args:
        sequences: An iterable of sequences to concatenate.
        axis: The axis along which to concatenate the sequences.

    Returns:
        The concatenated sequences.
    """
    arr_views = tuple(np.asarray(arr) for arr in sequences)
    seq = np.concatenate(arr_views, axis=axis, **kwargs).view(Sequence)

    arr_names = tuple(
        tuple(arr.names[dim] for arr in sequences) for dim in range(seq.ndim)
    )

    if axis < 0:
        axis = seq.ndim + axis

    names = []
    for dim, axis_names in enumerate(arr_names):
        unique_names = set(n for n in axis_names if n is not None)
        if len(unique_names) == 0:
            names.append(None)
            continue
        elif len(unique_names) > 1:
            raise ValueError(
                f"All names along axis {dim} must be the same. {arr_names[dim]}"
            )

        name = next(iter(unique_names))  # Get the axis name
        names.append(name)

        labels = []

        for s in sequences:
            label = s.labels.get(name)

            # append labels for each array to labels if it exists
            if label is not None:
                labels.append(label)
            # if labels is None, break if concatenation axis or continue otherwise
            elif dim == axis:
                break
        else:
            if dim == axis:
                seq.labels[name] = np.concatenate(labels)
            elif labels:
                unique_values = np.unique(np.stack(labels), axis=0)
                # Assign labels only if all labels along non-concatenation axis are the same
                if unique_values.shape[0] == 1:
                    seq.labels[name] = labels[0]

    seq.names = tuple(names)

    return seq


@sequence_implements(np.stack)
def stack(
    seqs: Sequence,
    axis=0,
    name: str | None = None,
    label: np.ndarray | None = None,
    **kwargs,
):
    """Joins sequences along a new axis.

    Args:
        sequences: An iterable of sequences to concatenate.
        axis: Specifies the new axis in the stacked sequences.
        name: A name for the new axis.
        label: Labels for the new axis. Must match the number of sequences.

    Returns:
        The joined sequences.
    """
    arr_views = tuple(np.asarray(arr) for arr in seqs)
    seq = np.stack(arr_views, axis=axis, **kwargs).view(Sequence)

    if axis < 0:
        axis = seq.ndim + axis

    names, labels = broadcast_names_and_labels(*seqs, raise_on_conflict=True)

    if name in names and name is not None:
        raise ValueError(f"Axis name {name} is already in names. {names}")
    else:
        names.insert(axis, name)

    if label is not None:
        if name is None:
            raise ValueError(f"Cannot add a label for an axis with no name.")

        if label.shape[0] != seq.shape[axis]:
            raise ValueError(
                f"Label has shape {label.shape} that is not compatible with "
                f"shape {seq.shape} on axis {axis}."
            )

        labels[name] = label

    seq.names = tuple(names)

    for n in seq.names:
        if (l := labels.get(n)) is not None:
            seq.labels[n] = l

    return seq


@sequence_implements(np.reshape)
def reshape(seq: Sequence, shape, **kwargs):
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
    axes=None,
):
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
def sum(seq: Sequence, axis: tuple[int, ...] | int | None = None, **kwargs):
    """Sum of sequence elements over a specified axis.

    Args:
        seq: The sequence to sum.
        axis: The axis or axes to add. If axis=None, the entire sequence
            is summed.

    Returns:
        The resulting sequence.
    """
    return seq.sum(axis, **kwargs)


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
