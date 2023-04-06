import functools
import re
import subprocess
import time
from subprocess import PIPE, STDOUT

import typer
from rich import print

app = typer.Typer(no_args_is_help=True)


def decode(bs: bytes) -> str:
    """Converts bytes to UTF-8 string.

    Replaces unicode escape codes with the corresponding unicode character. This is necessary because directly decoding the bytes with unicode_escape leads
    to double escaped characters.

    Args:
        bs: The bytes to convert to a unicode string.

    Returns:
        The converted unicode string.
    """
    s = bs.decode("utf-8")

    def convert(m):
        return m[0].encode("ascii").decode("unicode_escape")

    return re.sub(r"\\[Uu][0-9a-zA-Z]+", convert, s)


@app.command()
def format(
    ctx: typer.Context,
    check: bool = typer.Option(False, "--check/--no-check", "-c"),
):
    """Formats all source code.

    Runs isort followed by black.

    Args:
        check: If `True`, calls both black and isort with their respective check flags.
    """
    folders = ["src", "tests", "scripts"]

    isort = ["--check-only", *folders] if check else folders
    black = ["--check", *folders] if check else folders

    proc = [
        subprocess.run(["isort", *isort], stdout=PIPE, stderr=STDOUT),
        subprocess.run(["black", *black], stdout=PIPE, stderr=STDOUT),
    ]

    for p in proc:
        print(" ".join(p.args))
        print(decode(p.stdout))


@app.command()
def lint(
    ctx: typer.Context,
):
    ...


if __name__ == "__main__":
    app()
