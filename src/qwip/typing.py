from typing import (
    Annotated,
    Union,
    get_args,
    get_origin,
)
from numpy.typing import NDArray as _NDArray

NDArray = Annotated[_NDArray, '']

def is_annotated(tp):
    return get_origin(tp) is Annotated

def is_optional(tp):
    return get_origin(tp) == Union and type(None) in get_args(tp)