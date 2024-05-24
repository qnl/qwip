# Developer Guide

Welcome! This developer guide is meant to be a reference for the developers and maintainers of this library, helping to reduce maintenance burden. 

## Development Environment

To get started, you'll want to get your development environment set up with the proper tooling for making improvements to QWiP. 


<div class="termy">
```console
$ conda create -n qwip-env python=3.12
---> 100%
$ conda activate qwip-env
$ python -m pip install -e ".[dev]"
---> 100%
```

</div>

The `-e` flag will install QWiP in "editable" mode, so any changes made to the source code will be reflected when importing the library. The `[dev]` option will install all optional packages listed under `dev` in the `setup.cfg` file. These packages are needed for running tests, linting and formatting code, and building documentation. 

To verify that there are no issues with the installation, you can run the tests, all of which should pass.

<div class="termy">
```console
$ pytest 
---> 100%
```

</div>

If this checks out then you're all set and ready to go. You should read the remainder of this developer guide to familiarize yourself with best practices and guidelines on contributing changes to QWiP.

## Design Philosophy

*Flexible, with sane defaults.* This is the overarching principle behind QWiP's design, and should be kept in mind when adding new features and improvements to the library.

As experimental physics lab, we are always trying new things, some of which may not fit neatly into our original assumptions. This becomes cumbersome if our code is full of rigid assumptions that leave no room for future expansion. For example, you might assume that readout only ever occurs at the end of a sequence, that no one will ever want to readout the third (or fourth) level of transmon, or that AWG channels are always paired. While it may have been ok to hardcode these assumptions into our measurement libraries at some point in time, none of these are true now.

Of course we cannot anticipate all future changes, but what we can do is write code in a way that is composable and flexible.

On the other hand, a highly configurable and flexible system is inherently more complex than a system that is hard coded for a single task. To reduce the learning curve for new users and to make "standard" usage simple, there should always be sane defaults that allow the library to "just work".

*Maintainability* is another important principle to keep in mind. A majority of our time is spent reading old code rather than writing new code, so we should do what we can to make our future lives easier. This involves writing tests so that we can refactor implementation details with confidence and add new features without fear of breaking old use cases. This also includes writing documentation so that don't have to re-read old code and attempt to understand what we were thinking 8 months ago.