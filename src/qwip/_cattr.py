import datetime as dt
import re
from collections.abc import Mapping
from numbers import Number
from pathlib import Path
from pydoc import locate
from typing import Any, ForwardRef, TypeVar, get_args, get_origin

import attrs
import cattr
import numpy as np
import pendulum
from cattr import Converter
from cattr.gen import make_dict_structure_fn, make_dict_unstructure_fn
from loguru import logger
from numpy.typing import NDArray

converter = Converter(
    prefer_attrib_converters=True,
    unstruct_collection_overrides={set: list, frozenset: list},
)

# ========== builtin types ========== #
converter.register_structure_hook(
    bool, lambda v, cls: bool(v) if isinstance(v, (float, str, int)) else v
)

converter.register_structure_hook(
    Number, lambda v, cls: cls(v) if isinstance(v, Number) else v
)

converter.register_structure_hook(
    str, lambda v, cls: str(v) if isinstance(v, (int, float, bool)) else v
)

# ========== classes ========== #

CLS_REGEX = re.compile(r"(?P<cls>[^\[]*)(\[(?P<args>.*)\])")
ARGS_REGEX = re.compile(r",\s*(?![^\[\]]*\])")


def cls_to_string(cls):
    """Returns the fully qualified name for a class."""
    if get_args(cls):
        return str(cls)

    match cls.__module__:
        case "builtins":
            return cls.__name__
        case None:
            return cls.__name__
        case module:
            return ".".join((module, cls.__name__))


def locate_cls(cls_name):
    """Locates the class given a fully qualified name."""
    match = CLS_REGEX.match(cls_name)

    if match:
        cls_name, arg_names = match["cls"], match["args"]
    else:
        arg_names = None

    cls = locate(cls_name)
    if cls is None:
        raise NameError(f"name '{cls_name}' is not defined.")

    if arg_names is None:
        return cls

    args = converter.structure(ARGS_REGEX.split(arg_names), tuple[type, ...])

    return cls[args]


converter.register_unstructure_hook(type, lambda v: cls_to_string(v))

converter.register_structure_hook(type, lambda v, cls: locate_cls(v))

# ========== ForwardRef ========== #


def make_forward_ref_structure_fn(cls):
    def forward_ref_structure_fn(obj, cls):
        cls = cls._evaluate(None, None, set())
        return converter.structure(obj, cls)

    return forward_ref_structure_fn


def is_forward_ref(cls):
    return isinstance(cls, ForwardRef)


converter.register_structure_hook_factory(is_forward_ref, make_forward_ref_structure_fn)

# ========== TypeVar ========== #


def make_type_var_structure_fn(cls):
    def type_var_structure_fn(obj, cls):
        cls = cls.__bound__ or Any
        return converter.structure(obj, cls)

    return type_var_structure_fn


def is_type_var(cls):
    return isinstance(cls, TypeVar)


converter.register_structure_hook_factory(is_type_var, make_type_var_structure_fn)

# ========== stdlib types ========== #

converter.register_unstructure_hook(Path, lambda p: str(p))

converter.register_structure_hook(Path, lambda p, _: Path(p))


# ========== numpy types ========== #
def isndarray(cls):
    return issubclass(get_origin(cls) or cls, np.ndarray)


def get_dtype(cls):
    if not (args := get_args(cls)):
        return None

    _, dtypelike = args

    if dtypelike in get_args(NDArray):
        dtype = None
    else:
        (dtype,) = get_args(dtypelike)

    if dtype == Any:
        dtype = None

    return dtype


def make_ndarray_structure_fn(cls):
    dtype = get_dtype(cls)

    def ndarray_structure_fn(obj, cls):
        if isinstance(obj, np.ndarray):
            if dtype is None or obj.dtype == dtype:
                return obj
            else:
                try:
                    obj.dtype = dtype
                    return obj
                except ValueError:
                    pass  # Copy array if changing dtype in place fails
        return np.array(obj, dtype=dtype)

    return ndarray_structure_fn


def make_ndarray_unstructure_fn(cls):
    return lambda arr: arr.tolist()


converter.register_structure_hook_factory(isndarray, make_ndarray_structure_fn)

# Hook factory necesssary for unstructuring attrs classes with NDArray fields
converter.register_unstructure_hook_factory(isndarray, make_ndarray_unstructure_fn)

# ========== pendulum types ========== #


def pendulum_structure_fn(val, cls):
    if isinstance(val, str):
        return pendulum.parse(val)
    elif isinstance(val, dt.datetime):
        return pendulum.instance(val)

    return val


# Pendulum parse only accepts strings.
converter.register_structure_hook(pendulum.DateTime, pendulum_structure_fn)

converter.register_unstructure_hook(pendulum.DateTime, lambda dt: dt.isoformat())

converter.register_structure_hook(
    pendulum.Date, lambda v, cls: pendulum.parse(v).date()
)


converter.register_unstructure_hook(pendulum.Date, lambda dt: dt.isoformat())

# ========== attrs types ========== #


def make_attrs_structure_fn(cls, overrides: dict = {}):
    def get_override(field):
        if override := field.metadata.get("unstructure_override"):
            return override
        elif field.init is False:
            return cattr.override(omit=True)

    to_structure = {
        f.name: override for f in attrs.fields(cls) if (override := get_override(f))
    } | overrides

    structure_from_dict = make_dict_structure_fn(cls, converter, **to_structure)

    _cls = cls

    def structure_fn(v, cls):
        if isinstance(v, cls):
            return v
        else:
            return structure_from_dict(v, cls)

    return structure_fn


def make_attrs_unstructure_fn(cls, omit_defaults: bool = True, overrides: dict = {}):
    def get_override(field):
        if override := field.metadata.get("unstructure_override"):
            return override
        elif field.init is False and "serialize" not in field.metadata:
            return cattr.override(omit=True)
        elif field.metadata.get("serialize") is False:
            return cattr.override(omit=True)
        elif field.metadata.get("serialize") is True:
            return cattr.override(omit_if_default=False)

    to_unstructure = {
        f.name: override for f in attrs.fields(cls) if (override := get_override(f))
    } | overrides

    unstructure_from_dict = make_dict_unstructure_fn(
        cls, converter, _cattrs_omit_if_default=omit_defaults, **to_unstructure
    )

    def unstructure_fn(v):
        if v.__class__ is cls:
            return unstructure_from_dict(v)

        return converter.unstructure(v, unstructure_as=v.__class__)

    return unstructure_fn


converter.register_structure_hook_factory(attrs.has, make_attrs_structure_fn)

converter.register_unstructure_hook_factory(attrs.has, make_attrs_unstructure_fn)
