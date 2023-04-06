# Coding Guidelines

## Testing

Testing is important. Tests help guarantee that your code works as expected. They also serve as a way to clarify intended behavior. Tests ensure that changes made when adding features don't break existing features or use cases. Write tests.

We use pytest to run tests. See the pytest [documentation](https://docs.pytest.org/) for more information on how to use pytest. All tests should be in the `tests` folder, which mirrors the directory structure of the `src` folder.

## Code Style

We use [black](https://black.readthedocs.io/en/stable/the_black_code_style/index.html) and [isort](https://pycqa.github.io/isort) to autoformat code. This reduces time wasted on manual formatting and arguments over who's style is better. The important thing for legibility is that coding style is consistent to reduce distractions. Run the `scripts/format.py` script to autoformat your code prior to making committing your changes.

<div class="termy">
```console
$ python .\scripts\format.py format
isort src tests scripts
Fixing qwip\scripts\format.py

black src tests scripts
reformatted qwip\scripts\format.py

All done! ✨ 🍰 ✨
1 file reformatted, 88 files left unchanged.
$
```
</div>

## Linting

Linting will be done using [Ruff](https://beta.ruff.rs/docs/) but some work still needs to be done to bring existing code into compliance. New code should be checked with prior to committing.

There is also a VS Code extension for Ruff which should be installed.

