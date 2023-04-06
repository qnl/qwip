# Writing Documentation

Documentation is central to good software, making it easier to use and understand.

## Docstrings

Docstrings are important because they allow editors like jupyter and VS Code to quickly provide information about objects. They are also used to build the API docs.

All functions, methods, classes, and modules should have docstrings written using [google-style](https://google.github.io/styleguide/pyguide.html#381-docstrings) format. Function and method arguments should use python type annotations to denote the expected types rather than placing these in the docstring.

## Extended Documentation

But while docstrings are helpful, they are often insufficient for new users or providing a solid conceptual understanding of how to properly use the library. To facilitate learning, we provide example-based documentation and explanations of key concepts and modules in the user guide.

## MkDocs

We use [MkDocs](https://www.mkdocs.org/) and the fantastic [Material for MkDocs](https://www.mkdocs.org/) theme to build the documentation pages. 

When making changes to the documentation, you can build the documentation locally by running `mkdocs serve` from the `docs` directory. This will build the documentation and spin up an http server to host the documentation site locally. This also watches all files for changes and will rebuild the documentation when any files are updated.

<div class="termy">
```console
$ cd docs
$ mkdocs serve
INFO     -  Building documentation...
INFO     -  Cleaning site directory
INFO     -  Documentation built in 2.03 seconds
INFO     -  [16:15:54] Watching paths for changes: 'pages', 'mkdocs.yml', 'src\qwip'
INFO     -  [16:15:54] Serving on http://127.0.0.1:8000/QWiP/
```
</div>