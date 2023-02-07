# Introduction

Configuration management is a core component of the measurement software stack. Running a quantum hardware experiment requires keeping track of large number of constantly changing parameters, including qubit frequencies, pulse amplitudes, and readout discrimination boundaries among others. To effectively manage the ever-growing collection of parameters effectively, QWiP provides a version-controlled database for managing the *configuration* of our quantum hardware systems.

## ConfigDB

QWiP handles the task of managing configuration parameters using [Dolt](https://docs.dolthub.com/), which is a MySQL compatible database that supports git-like versioning of tables and data. This allows parameters to be persisted to disk while also supporting version control of the system configuration. 

Database interactions are simplified via a ConfigDB object, which provides a dictionary-like interface for accessing parameters, querying history, and performing version-control operations on the database tables.


## Why not YAML?

Historically, system configuration was persisted to disk via a set of YAML files. While simple, this implementation came with several shortcomings that QWiP aims to address.

1. **Direct file-based persistence does not ensure data integrity.**

    Data integrity is important when dealing with a large number of system parameters that takes a significant amount of time to calibrate. With file-based storage, complete data loss or corruption can occur if your code is interupted while writing the file to disk or when multiple processes attempt to write to the same file simultaneously.

2. **No persistence of historical parameter values.**

    The old YAML-based configuration system forces you to overwrite old parameter values each time a parameter is updated. While one could in principle version control these yaml files with git, it would stil be difficult to query the parameter value over the entire commit history.




