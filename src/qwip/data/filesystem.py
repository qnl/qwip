import re
from pathlib import Path
from typing import Callable
from uuid import uuid4

import pendulum

from qwip.defaults import dynamic_default
from qwip.flatdict import FlatDict

DIRECTORY_RULES: dict[str, Callable] = FlatDict()
FILENAME_RULES: dict[str, Callable] = FlatDict()

camel2kebab_1 = re.compile(r"(.)([A-Z][a-z]+)")
camel2kebab_2 = re.compile(r"([a-z0-9])([A-Z])")


def add_extension(name: str, ext: str) -> str:
    """Add a file extension to a filename.

    If the filename already ends with the specified extension, no changes are made.

    Args:
        name: A filename.
        ext: An extension string. An extension can be specified with or without the '.'.
            For example, `.csv` or `csv` are both valid and will yield the same result.

    Returns:
        A modified filename with the specified extension.
    """
    ext = ext if ext.startswith(".") or ext == "" else f".{ext}"
    name = name if name.endswith(ext) else f"{name.rstrip('.')}{ext}"

    return name


def camel_to_kebab(name: str) -> str:
    """Converts a `CamelCase` string to a `kebab-case` string.

    See [StackOverflow post](https://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-snake-case)
    for reference.

    Args:
        name: The string to convert.

    Returns:
        The converted string.
    """
    name = camel2kebab_1.sub(r"\1-\2", name)
    return camel2kebab_2.sub(r"\1-\2", name).lower()


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
    **kwargs,
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


__all__ = ["add_extension", "camel_to_kebab"]
