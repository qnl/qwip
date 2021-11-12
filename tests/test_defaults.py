import functools
import inspect

import pytest
import attr

from loguru import logger

from qwip import qsettings
from qwip.settings.settings import Settings, qattrs
from qwip.defaults import QWiPDefault, dynamic_default


@qattrs
class ChildArgs(Settings):
    default_int: int = 2
    default_float: float = 0.2
    default_str: str = 'child'

@qattrs
class DefaultArgs(Settings):
    default_int: int = 1
    default_float: float = 0.1
    default_str: str = 'parent'
    child: ChildArgs = attr.ib(factory=ChildArgs)

@pytest.fixture()
def defaults():
    return DefaultArgs()

def test_replace_pos_or_kw_nodefaults(defaults):
    @dynamic_default(__settings__=defaults, arg1='default_int', arg2='default_float', arg3='default_str')
    def echo_args(arg1, arg2, arg3):
        return arg1, arg2, arg3

    signature = inspect.signature(echo_args)

    params = list(signature.parameters)
    defaults = [param.default.path for param in signature.parameters.values()]
    
    assert params == ['arg1', 'arg2', 'arg3']
    assert defaults == ['default_int', 'default_float', 'default_str']

    for param in signature.parameters.values():
        assert param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
        assert isinstance(param.default, QWiPDefault)

def test_replace_pos_or_kw_defaults(defaults):
    @dynamic_default(__settings__=defaults, kwarg1='default_int', kwarg2='default_float', kwarg3='default_str')
    def echo_kwargs(kwarg1=None, kwarg2=None, kwarg3=None):
        return kwarg1, kwarg2, kwarg3

    signature = inspect.signature(echo_kwargs)

    params = list(signature.parameters)
    defaults = [arg.default.path for arg in signature.parameters.values()]
    
    assert params == ['kwarg1', 'kwarg2', 'kwarg3']
    assert defaults == ['default_int', 'default_float', 'default_str']

    for param in signature.parameters.values():
        assert param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
        assert isinstance(param.default, QWiPDefault)

def test_replace_kwonly(defaults):
    @dynamic_default(__settings__=defaults, kwarg1='default_int', kwarg2='default_float')
    def echo_kwonly(*, kwarg1=None, kwarg2=None):
        return kwarg1, kwarg2

    signature = inspect.signature(echo_kwonly)

    params = list(signature.parameters)
    defaults = [param.default.path for param in signature.parameters.values()]
    
    assert params == ['kwarg1', 'kwarg2']
    assert defaults == ['default_int', 'default_float']

    for param in signature.parameters.values():
        assert param.kind == inspect.Parameter.KEYWORD_ONLY
        assert isinstance(param.default, QWiPDefault)

def test_replace_positional_only(defaults):
    with pytest.raises(ValueError):
        @dynamic_default(__settings__=defaults, arg1='default_int', arg2='default_float')
        def echo_posonly(arg1, arg2, /):
            return arg1, arg2

def test_replace_nonexistent_parameter(defaults):
    with pytest.raises(ValueError):
        @dynamic_default(__settings__=defaults, nonexistent_param='default_int')
        def echo_args(arg):
            return arg

def test_replace_nonexistent_key(defaults):
    with pytest.raises(ValueError):
        @dynamic_default(__settings__=defaults, arg='nonexistent_key')
        def echo_args(arg):
            return arg

@pytest.fixture
def echo_function(defaults):
    @dynamic_default(__settings__=defaults,
        arg1='default_int',
        kwarg2='default_str',
        kwarg3='child/default_str')
    def fn(arg, /, arg1, kwarg1=None, kwarg2=None, *, kwarg3=None):
        return arg, arg1, kwarg1, kwarg2, kwarg3

    return fn

def test_default_values(echo_function):
    assert echo_function('arg') == ('arg', 1, None, 'parent', 'child')

def test_pass_by_position(echo_function):
    assert echo_function('arg', 2, None, 'new_parent') == ('arg', 2, None, 'new_parent', 'child')

def test_pass_by_keyword(echo_function):
    assert echo_function('arg', arg1=2) == ('arg', 2, None, 'parent', 'child')
    assert echo_function('arg', kwarg2='new_parent', kwarg3='new_child') == ('arg', 1, None, 'new_parent', 'new_child')

def test_change_defaults(defaults):
    @dynamic_default(__settings__=defaults,
        arg1='default_int',
        kwarg2='default_str',
        kwarg3='child/default_str')
    def echo_function(arg, /, arg1, kwarg1=None, kwarg2=None, *, kwarg3=None):
        return arg, arg1, kwarg1, kwarg2, kwarg3

    assert echo_function('arg') == ('arg', 1, None, 'parent', 'child')
    
    defaults['default_int'] = 2
    defaults['child/default_str'] = 'new_child'
    assert echo_function('arg') == ('arg', 2, None, 'parent', 'new_child')

def test_default_repr(echo_function):
    params = inspect.signature(echo_function).parameters

    assert repr(params['arg1'].default) == 'QWiPDefault(default_int=1)'
    assert repr(params['kwarg2'].default) == "QWiPDefault(default_str='parent')"

 