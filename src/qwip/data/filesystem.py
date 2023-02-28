from pathlib import Path
from typing import Callable, Dict
from uuid import uuid4

import pendulum
from loguru import logger

from qwip.defaults import dynamic_default
from qwip.flatdict import FlatDict

DIRECTORY_RULES: Dict[str, Callable] = FlatDict()
FILENAME_RULES: Dict[str, Callable] = FlatDict()


@dynamic_default(base="data/base_directory", rule="data/directory_rule")
def get_data_directory(base: str = None, dirname: str = "", rule: str = None, **kwargs):
    """ """
    if not dirname:
        dirname = DIRECTORY_RULES[rule](**kwargs)

    base = Path(base).resolve()
    directory = base / dirname

    return directory


@dynamic_default(
    base="data/base_directory",
    rule="data/directory_rule",
    exist_ok="data/directory_exist_ok",
)
def make_data_directory(
    base: str = None,
    dirname: str = "",
    rule: str = None,
    exist_ok: bool = None,
    **kwargs
):
    directory = get_data_directory(base, dirname, rule, **kwargs)
    directory.mkdir(parents=True, exist_ok=exist_ok)

    return directory


def get_rules():
    return list(DIRECTORY_RULES.keys())


def directory_rule(func):
    DIRECTORY_RULES[func.__name__] = func
    return func


@directory_rule
@dynamic_default(date_fmt="data/date_fmt")
def date(name_fmt: str = "{date}", date_fmt: str = None):
    date = pendulum.now().format(date_fmt)
    return name_fmt.format(date=date)


@directory_rule
def uuid(name_fmt: str = "{uuid}"):
    return name_fmt.format(uuid=uuid4())


@directory_rule
def timestamp(name_fmt: str = "{timestamp}"):
    return name_fmt.format(timestamp=pendulum.now().int_timestamp)
