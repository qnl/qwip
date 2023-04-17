# QWiP

Quantum Workflows in Python (QWiP) is a framework for running experiments with superconducting qubit devices. To get started using this library follow the installation instructions below, and then check out the [User Guide](./user-guide/).

## Installation

To install QWiP, first clone the GitHub [repository](https://github.com/qnl/QWiP). Then create a new conda environment with your preferred version of python, keeping in mind that QWiP supports python 3.10 and up.

<div class="termy">
```console
$ conda create -n qwip-env python=3.10
---> 100%
$ conda activate qwip-env
$ python -m pip install .
---> 100%
```
</div>

If you plan to do additional development with QWiP, see the developer guide for alternative setup instructions on how to create your development environment.

## Contributing

There are many ways to contribute to making QWiP a better experience for all. This includes adding new features, fixing bugs, increasing test coverage, and improving documentation. To get started, check out the [Developer Guide](./developer-guide/).

