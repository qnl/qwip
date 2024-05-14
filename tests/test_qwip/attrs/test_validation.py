from collections.abc import Callable, Iterable, Mapping
from contextlib import nullcontext as does_not_raise
from typing import Annotated, Any, Optional, Self, Union

import attrs
import numpy as np
import pytest
from attr import define, field
from attr.exceptions import NotCallableError
from numpy.typing import NDArray

from qwip.attrs import qdefine
from qwip.attrs.validation import get_type_validator, resolve_types_with_validation
from qwip.flatdict import FlatDict


class TestGetTypeValidators:
    def test_any(self):
        assert get_type_validator(Any) is None

    @pytest.mark.parametrize(
        "tp,value,expect",
        [
            (str, "string", does_not_raise()),
            (str, 1, pytest.raises(TypeError)),
            (int, 1, does_not_raise()),
            # bool is subclasses of int
            (int, True, does_not_raise()),
            # Simple iterables
            (Iterable, {}, does_not_raise()),
            (list, [], does_not_raise()),
            (list, 10, pytest.raises(TypeError)),
            # Simple Mappings
            (Mapping, {}, does_not_raise()),
            (dict, 10, pytest.raises(TypeError)),
            # Unions
            (int | str, 1, does_not_raise()),
            (Union[int, str], "string", does_not_raise()),
            (int | str, list(), pytest.raises(TypeError)),
            # Optional
            (int | None, 1, does_not_raise()),
            (int | None, None, does_not_raise()),
            (Optional[str], "string", does_not_raise()),
            (Optional[str], list(), pytest.raises(TypeError)),
            (Union[None, int, str], None, does_not_raise()),
        ],
    )
    def test_simple(self, tp, value, expect):
        validate = get_type_validator(tp)

        @define
        class A:
            attr: tp = field(validator=validate)

        with expect:
            A(attr=value)

    @pytest.mark.parametrize(
        "tp,value,expect",
        [
            (Annotated[str, "annotation"], "string", does_not_raise()),
            (Annotated[str | int, "annotation"], 1, does_not_raise()),
            (Annotated[Optional[str], "annotation"], None, does_not_raise()),
            (Annotated[Optional[str], "annotation"], 1, pytest.raises(TypeError)),
        ],
    )
    def test_annotated(self, tp, value, expect):
        validate = get_type_validator(tp)

        @define
        class A:
            attr: tp = field(validator=validate)

        with expect:
            A(attr=value)

    @pytest.mark.parametrize(
        "tp,value,expect",
        [
            (list[Any], [], does_not_raise()),
            (list[Any], 10, pytest.raises(TypeError)),
            (set[str], {"string"}, does_not_raise()),
            (set[str], set(), does_not_raise()),
            (set[str], ["string"], pytest.raises(TypeError)),
        ],
    )
    def test_iterable(self, tp, value, expect):
        validate = get_type_validator(tp)

        @define
        class A:
            attr: tp = field(validator=validate)

        with expect:
            A(attr=value)

    @pytest.mark.parametrize(
        "tp,value,expect",
        [
            (dict[Any], {}, does_not_raise()),
            (dict[Any, Any], {}, does_not_raise()),
            (dict[Any, Any], 10, pytest.raises(TypeError)),
            (dict[Any, str], {1: "string", True: "string"}, does_not_raise()),
            (dict[Any, str], {1: "string", True: 1}, pytest.raises(TypeError)),
            (dict[str], {}, does_not_raise()),
            (dict[str], {"a": 1}, does_not_raise()),
            (dict[str, Any], {1: "a"}, pytest.raises(TypeError)),
            (Mapping[str, int | float], {"a": 1, "b": 1.5}, does_not_raise()),
            (Mapping[str, int | float], {"a": 1, "b": "c"}, pytest.raises(TypeError)),
            (
                FlatDict[str, int | FlatDict],
                FlatDict({"a/b/c": 1, "a/d": 2, "e": 3}),
                does_not_raise(),
            ),
        ],
    )
    def test_mapping(self, tp, value, expect):
        validate = get_type_validator(tp)

        @define
        class A:
            attr: tp = field(validator=validate)

        with expect:
            A(attr=value)

    @pytest.mark.parametrize(
        "tp,value,expect",
        [
            (dict[str, list[int]], {}, does_not_raise()),
            (dict[str, list[int]], {"a": [], "b": [1, 2]}, does_not_raise()),
            (
                dict[str, dict[int, int]],
                {"a": [], "b": [1, 2]},
                pytest.raises(TypeError),
            ),
            (
                frozenset[tuple[str, int]],
                frozenset({(1, 1), (2, 2)}),
                pytest.raises(TypeError),
            ),
        ],
    )
    def test_nested_containers(self, tp, value, expect):
        validate = get_type_validator(tp)

        @define
        class A:
            attr: tp = field(validator=validate)

        with expect:
            A(attr=value)

    @pytest.mark.parametrize(
        "tp,value,expect",
        [
            (Optional[dict[str, int]], {}, does_not_raise()),
            (Optional[dict[str, int]], None, does_not_raise()),
            (dict[str, int] | None, {"a": []}, pytest.raises(TypeError)),
        ],
    )
    def test_optional(self, tp, value, expect):
        validate = get_type_validator(tp)

        @define
        class A:
            attr: tp = field(validator=validate)

        with expect:
            A(attr=value)

    @pytest.mark.parametrize(
        "tp,value,expect",
        [
            (int | dict[str, int], {}, does_not_raise()),
            (list[int] | dict[str, int], {}, does_not_raise()),
            (int | dict[str, int], "string", pytest.raises(TypeError)),
            (int | dict[str, int | dict], dict(test=dict()), does_not_raise()),
        ],
    )
    def test_union(self, tp, value, expect):
        validate = get_type_validator(tp)

        @define
        class A:
            attr: tp = field(validator=validate)

        with expect:
            A(attr=value)

    @pytest.mark.parametrize(
        "tp,value,expect",
        [
            (NDArray, np.array([1]), does_not_raise()),
            (NDArray, np.array([1], dtype=np.float64), does_not_raise()),
            (NDArray, np.array([1], dtype=np.complex128), does_not_raise()),
            (NDArray[np.float64], np.array([1.5])[0], pytest.raises(TypeError)),
            (NDArray[np.float64], np.array([1], dtype=np.float64), does_not_raise()),
            (NDArray[np.float64], np.array([1]), pytest.raises(TypeError)),
        ],
    )
    def test_numpy(self, tp, value, expect):
        validate = get_type_validator(tp)

        @define
        class A:
            attr: tp = field(validator=validate)

        with expect:
            A(attr=value)

    @pytest.mark.parametrize(
        "tp,value,expect",
        [
            (tuple, (1, 2, 3), does_not_raise()),
            (tuple[int, ...], (1, 2, 3), does_not_raise()),
            (tuple[int, ...], ("a", "b", 3), pytest.raises(TypeError)),
            (tuple[int], (1, 2, 3), pytest.raises(TypeError)),
            (tuple[int, str, bool], (1, "b", True), does_not_raise()),
            (
                tuple[int | str, Optional[bool], list[float]],
                (1, None, [2.0]),
                does_not_raise(),
            ),
            (
                tuple[int | str, Optional[bool], list[float]],
                (1, None, ["string"]),
                pytest.raises(TypeError),
            ),
        ],
    )
    def test_tuple(self, tp, value, expect):
        validate = get_type_validator(tp)

        @define
        class A:
            attr: tp = field(validator=validate)

        with expect:
            A(attr=value)

    @staticmethod
    def func1(x, y): ...

    @staticmethod
    def func2(x: int) -> str:
        return "a"

    @pytest.mark.parametrize(
        "tp,value,expect",
        [
            (Callable, func1, does_not_raise()),
            (Callable[[int], str], func2, does_not_raise()),
            (Callable[..., str], 1, pytest.raises(NotCallableError)),
        ],
    )
    def test_callable(self, tp, value, expect):
        validate = get_type_validator(tp)

        @define
        class A:
            attr: tp = field(validator=validate)

        with expect:
            A(attr=value)


class TestAddTypeValidator:
    @pytest.mark.parametrize(
        "tp,value,expect",
        [
            (int, 0, does_not_raise()),
            (int, "string", pytest.raises(TypeError)),
        ],
    )
    def test_add_type_validators(self, tp, value, expect):
        @qdefine
        class A:
            attr: tp

        with expect:
            a = A(attr=value)

    @pytest.mark.parametrize(
        "tp,value,expect",
        [
            (tuple, 1, does_not_raise()),
            (int | str, list(), does_not_raise()),
            (int | dict[str, int], "string", does_not_raise()),
            (list[int], 1, does_not_raise()),
        ],
    )
    def test_validate_metadata(self, tp, value, expect):
        @qdefine
        class A:
            attr: tp = field(metadata=dict(validate=False))

        with expect:
            a = A(attr=value)

        assert a.attr == value

    def test_extra_validators(self):
        from attrs.validators import le

        @qdefine
        class A:
            attr: int = field(validator=le(0))

        with pytest.raises(TypeError):
            A(attr="string")

        with pytest.raises(ValueError):
            A(attr=1)

    def test_self_type(self):
        @qdefine
        class A:
            attr: Self

        field = attrs.fields(A).attr
        assert field.type._evaluate(globals(), locals(), set()) == A

    def test_optional_nullvalidator(self):
        @qdefine
        class A:
            attr: Self | None

        field = attrs.fields(A).attr
        assert field.validator is None


def test_resolve_types_with_validation():
    @qdefine
    class Tree:
        left: "Tree | None" = None
        right: "Optional[Tree]" = None
        value: "str" = "tree"

    t = Tree(value=1)
    # No validator added for forward references
    assert t.value == 1

    resolve_types_with_validation(Tree, globalns=globals(), localns=locals())

    t1 = Tree()
    t2 = Tree()

    t3 = Tree(left=t1, right=t2)

    assert t3.left == t1 and t3.right == t2

    # Now forward references have been resolved and validators added
    with pytest.raises(TypeError):
        t1.left = 1
