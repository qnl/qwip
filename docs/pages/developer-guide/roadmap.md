# Roadmap

QWiP is still in the early stages of development, with many new features planned with the goal of making it easier to set up and execute new experiments.

## QPU Servers

This eliminates setup "bookings" except for long data taking sessions. Multiple people can take data from different client computers, by sending sequences to a single server hooked up to the measurement hardware. The simplest implementation involves putting most of the logic client side and leaving only the final sequence upload and data acquisition to the server. Server sends raw data back to the client and remainder of processing is done client side.

Nice to have features would include easy deployment/upgrade scripts for updating server functionality, and monitoring to ensure system stability.

## Flexible Processing

Sometimes I want three state readout. Sometimes I want two state readout without losing fidelity to trying to classify the third state. Sometimes you want excited state promotion. Often I find myself wanting to switch between a different automated data processing configurations without having to overwrite my configuration yaml or creating a separate config file. Processing (and automation in general) should be flexible enough to allow for such context switching.

Also result data should be structured and not a random dictionary.

## SystemDB

Much of the complexity of running a mulitqubit device comes from maintaining the large number of system parameters needed to run a circuit/experiment on the device. This includes information such as qubit frequencies, pulse parameters, and readout discrimination data. 

Sometimes we want to switch temporarily to a different set of configuration parameters, and often we want to compare current calibrated parameters to old calibrated parameters. It is also helpful to have some sort of gui to explore configuration parameters in a human readable way.

To do this properly we need to move beyond file based storage and move configuration data into a database. Ideally we also have first class support for version management and branching.

[Dolt](https://docs.dolthub.com/introduction/use-cases/config) seems to be a promising candidate for the SystemDB. It is a mysql-compatible database that acts like "git for data". Another candidate is [TerminusDb](https://terminusdb.com/docs/get-started/index) which is a document-based database.