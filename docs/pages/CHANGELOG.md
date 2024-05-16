# Changelog

Changes to QWiP will be documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to the [Calendar Versioning](https://calver.org/) convention.

## [Unreleased]

### Added

- Automated batching by timeline or repetition in the QPU.

### Fixed

- Fixed restrictions on adding more than one loop in the data processor dependency graph.

## [24.05.0] - 2024-05-13

### Added

- New data management features for auto-saving data and searching
- New backends for Tektronix 5014C AWG, Alazar ADC, Qubic.
- Resonator fitting models for reflection/hanger geometries.
- Generic data processors for labeling/averaging results.

### Changed

- Upgraded supported python version to 3.12
- Renamed `SequenceElement` to `Timeline`.
- Renamed `ModulationFrequency` to `Frame` to align more closely with OpenQASM terminology.

