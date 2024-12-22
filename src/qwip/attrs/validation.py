from collections.abc import Callable, Iterable, Mapping
from functools import reduce
from types import GenericAlias, UnionType
from typing import (
    Annotated,
    Any,
    ForwardRef,
    Self,
    Type,
    TypeVar,
    Union,
    _SpecialForm,
    get_args,
    get_origin,
)

import attrs
import numpy as np
from attr._make import _OBJ_SETATTR
from attrs import field, frozen, resolve_types
from attrs.validators import (
    and_,
    deep_iterable,
    deep_mapping,
    instance_of,
    is_callable,
    optional,
)
from loguru import logger
from numpy.typing import NDArray

from qwip.typing import replace_self_type, typedispatch

ScalarType = get_args(get_args(NDArray)[1])[0]


@frozen(repr=False)
class _NullValidator:
    def __call__(self, inst, attr, value):
        """
        We use a callable class to be able to change the ``__repr__``.
        """
        pass

    def __repr__(self):
        return "<Null validator>"


@frozen(repr=False)
class _NumpyTypeValidator:
    dtype = field()

    def __call__(self, inst, attr, value):
        """
        We use a callable class to be able to change the ``__repr__``.
        """

        valid_dtype = self.dtype == ScalarType or value.dtype == self.dtype
        if not (isinstance(value, np.ndarray) and valid_dtype):
            raise TypeError(
                "'{name}' must be a {type!r} with dtype {dtype!r} "
                "(got {value!r} that is a {actual!r}){optional_dtype}.".format(
                    name=attr.name,
                    type=np.ndarray,
                    dtype=self.dtype,
                    actual=value.__class__,
                    value=value,
                    optional_dtype=(
                        f" with dtype {value.dtype}"
                        if isinstance(value, np.ndarray)
                        else ""
                    ),
                ),
                attr,
                self.dtype,
                value,
            )

    def __repr__(self):
        return "<Numpy validator>"


@frozen(repr=False)
class _DeepTupleValidator:
    member_validators = field(validator=deep_iterable(optional(is_callable())))
    tuple_validator = field(default=None, validator=optional(is_callable()))

    def __call__(self, inst, attr, value):
        """
        We use a callable class to be able to change the ``__repr__``.
        """

        if self.tuple_validator is not None:
            self.tuple_validator(inst, attr, value)

        if len(self.member_validators) != len(value):
            raise TypeError(
                "'{name}' must be a {type!r} with length {length!r} "
                "(got {value!r} with length {actual!r}).".format(
                    name=attr.name,
                    type=attr.type,
                    length=len(self.member_validators),
                    value=value,
                    actual=len(value),
                ),
                attr,
                attr.type,
                value,
            )

        for validator, member in zip(self.member_validators, value):
            if validator:
                validator(inst, attr, member)

    def __repr__(self):
        return "<Tuple validator>"


@frozen(slots=True)
class _UnionValidator:
    """
    Tries all validators and only raises type error if all validators do.
    """

    _validators = field()

    def __call__(self, inst, attr, value):
        """
        We use a callable class to be able to change the ``__repr__``.
        """
        for v in self._validators:
            try:
                v(inst, attr, value)
                break
            except TypeError:
                pass
        else:
            raise TypeError(
                "'{name}' must be one of {type!r} (got {value!r} that is a "
                "{actual!r}).".format(
                    name=attr.name,
                    type=attr.type,
                    actual=value.__class__,
                    value=value,
                ),
                attr,
                attr.type,
                value,
            )


@typedispatch
def get_validator(tps: Type | tuple[Type], cls: type | None = None) -> Callable:
    raise NotImplementedError(f"Type validator for {tps} is not implemented.")


@get_validator.register(Any)
def get_any_validator(tps: Type, cls: type | None = None) -> Callable:
    return None


@get_validator.register(Self)
def get_self_validator(tps: Type, cls: type | None = None) -> Callable:
    return get_type_validator(ForwardRef(cls.__name__), cls)


@get_validator.register(Annotated)
def get_annotated_validator(tps: Type, cls: type | None = None) -> Callable:
    return get_type_validator(get_args(tps)[0], cls)


@get_validator.register(Callable)
def get_annotated_validator(tps: Type, cls: type | None = None) -> Callable:
    return is_callable()


@get_validator.register(tuple)
def get_tuple_validator(tps: Type, cls: type | None = None) -> Callable:
    args = get_args(tps)
    tuple_validator = get_type_validator(get_origin(tps), cls)

    if ... in args:
        if len(args) != 2:
            raise ValueError(f"{tps} is not a sensible type.")

        member_validator = get_type_validator(args[0], cls)
        if member_validator:
            return deep_iterable(member_validator, tuple_validator)
        else:
            return tuple_validator

    else:
        validators = tuple(get_type_validator(a, cls) for a in args)
        return _DeepTupleValidator(validators, tuple_validator)


@get_validator.register(np.ndarray)
def get_numpy_validator(tps: Type, cls: type | None = None) -> Callable:
    _, dtype = get_args(tps)

    dtype = get_args(dtype)[0]
    return _NumpyTypeValidator(dtype=dtype)


@get_validator.register(Mapping)
def get_mapping_validator(tps: Type, cls: type | None = None) -> Callable:
    args = get_args(tps)
    key_validator = get_type_validator(args[0], cls) or _NullValidator()
    mapping_validator = get_type_validator(get_origin(tps), cls)

    value_validator = None
    if len(args) > 1:
        value_validator = get_type_validator(args[1], cls)

    value_validator = value_validator or _NullValidator()

    # Necessary to handle pathological case where mapping[Any, Any] is used
    if isinstance(key_validator, _NullValidator) and isinstance(
        value_validator, _NullValidator
    ):
        return mapping_validator
    else:
        return deep_mapping(key_validator, value_validator, mapping_validator)


@get_validator.register(Iterable)
def get_iterable_validator(tps: Type, cls: type | None = None) -> Callable:
    member_validator = get_type_validator(get_args(tps), cls)
    iterable_validator = get_type_validator(get_origin(tps), cls)

    # Necessary to handle pathological case where list[Any] is used
    if member_validator:
        return deep_iterable(member_validator, iterable_validator)
    else:
        return iterable_validator


def get_optional_validator(tps: Type, cls: type | None = None) -> Callable:
    args = tuple(tp for tp in get_args(tps) if tp is not type(None))

    validator = get_type_validator(args, cls)
    return optional(validator) if validator else None


@get_validator.register(Union)
@get_validator.register(UnionType)
def get_union_validator(tps: Type, cls: type | None = None) -> Callable:
    args = get_args(tps)

    if type(None) in args:
        return get_optional_validator(tps, cls)

    def is_special_type(tp):
        """Determines if a type can be directly handled within a union"""
        special_types = (_SpecialForm, GenericAlias, UnionType, ForwardRef, str)

        return isinstance(tp, special_types) or get_origin(tp) or tp == ...

    combined = tuple(tp for tp in args if not is_special_type(tp))
    special = (tp for tp in args if is_special_type(tp))

    validators = []

    if combined:
        validators += [get_type_validator(reduce(lambda a, b: a | b, combined), cls)]

    for tp in special:
        validators += [get_type_validator(tp, cls)]

    return _UnionValidator(validators)


def get_tuple_of_types_validator(tps: tuple[Type], cls: type | None = None) -> Callable:
    if len(tps) == 0:
        return None
    elif len(tps) == 1:
        return get_type_validator(tps[0], cls)
    else:
        return get_union_validator(reduce(lambda a, b: a | b, tps), cls)


def get_type_validator(tps: Type | tuple[Type], cls: type | None = None):
    try:
        isinstance(None, tps)
        return instance_of(tps)
    except TypeError:
        pass

    if isinstance(tps, tuple):
        # Need to filter out special types
        return get_tuple_of_types_validator(tps, cls)
    elif isinstance(tps, (str, ForwardRef)):
        logger.info(f"No validator added for forward reference {repr(tps)}.")
        return None
    elif isinstance(tps, TypeVar):
        if tps.__bound__:
            return get_type_validator(tps.__bound__, cls)
        return None
    else:
        return get_validator(tps, cls)


def add_type_validators(cls, fields):
    new_fields = []

    for field in fields:
        if not field.metadata.get("validate", True):
            new_fields.append(field)
            continue

        type_validator = get_type_validator(field.type, cls)
        updated_type = replace_self_type(field.type, cls)

        if type_validator is None:
            if field.type == updated_type:
                new_fields.append(field)
            else:
                new_fields.append(field.evolve(type=updated_type))
            continue

        if field.validator is not None:
            # Put type validation before additional validators
            type_validator = and_(type_validator, field.validator)

        new_fields.append(field.evolve(validator=type_validator, type=updated_type))

    return new_fields


from functools import wraps


def resolve_types_with_validation(maybe_cls=None, globalns=None, localns=None):
    def wrapper(cls):
        forward_ref_fields = [
            f.name for f in attrs.fields(cls) if isinstance(f.type, str)
        ]
        cls = resolve_types(cls, globalns=globalns, localns=localns)

        # Now we add validators that weren't previously added
        for field in attrs.fields(cls):
            # Don't add a duplicate validator if one was already added
            # Or if auto-validation is turned off.
            if not (
                field.name in forward_ref_fields
                and field.metadata.get("validate", True)
            ):
                continue

            type_validator = get_type_validator(field.type, cls)
            if type_validator is None:
                continue

            if field.validator is not None:
                type_validator = and_(type_validator, field.validator)

            _OBJ_SETATTR(field, "validator", type_validator)

        return __build_class__

    if maybe_cls is None:
        return wrapper
    else:
        return wrapper(maybe_cls)
