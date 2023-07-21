import pytest

from qwip.testing import ignore_order


class TestUnorderedList:
    TEST_CASES = [
        (["a", "b", "c"], ignore_order(["a", "b", "c"]), True),
        (("a", "b", "c"), ignore_order(("c", "b", "a")), True),
        ((char for char in "abc"), ignore_order((char for char in "cba")), True),
        ([{"a": 1}, {"b": 2}], ignore_order([{"b": 2}, {"a": 1}]), True),
        (["a", "b", "c"], ignore_order(["c"]), False),
        (["a", "b", "c"], ignore_order(["c", "c", "c"]), False),
        (["a", "b", "c"], ignore_order(("a", "b", "c")), False),
        (["a", "b", "c"], ignore_order(("a", "b", "c"), ignore_type=True), True),
    ]

    @pytest.mark.parametrize("a,b,expect", TEST_CASES)
    def test_compare(self, a, b, expect):
        assert (a == b) is expect
