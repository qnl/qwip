# Changelog

Changes to QWiP will be documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to the [Calendar Versioning](https://calver.org/) convention.

## [Unreleased]

### Added

- Automated batching by timeline or repetition in the QPU.
- Waveform convolutions using the multiplication operator.
- Waveform channels can now be reassigned at the waveform or timeline level. This makes it possible to add a single pulse prototype for pulses on different channels/qubits.
- Default plotting methood for visualization waveforms.

### Changed

- Change sequence labels from numpy arrays to pandas index to allow more flexible labeling.
- `Waveforms` can now only contain a single channel.
- Timeline location variables and waveform variables are now sympy symbols/expressions. This allows for a shared set of variables as well as nonlinear expressions to be used.
- Quantum systems no longer track local oscillator frames directly. Instead, local oscillator frequencies are added as separate frames to the compiler. When doing IQ mixing or some other form of upconversion, specify the waveform frequency as `freq - lo_freq`.
- The readout configuration schema in `ConfigDB` has been modified.

### Fixed

- Fixed restrictions on adding more than one loop in the data processor dependency graph.
- Fixed bug in `QPU.save_compiler` where compiler class was not getting saved.

### Deprecated
- `Timeline.resolve_waveforms` and `Timeline.resolve_locations` are now deprecated because waveform variables and location variables are no longer treated separately, use `Timeline.resolve` instead.

## [24.05.1] - 2024-05-13

### Added

- New data management features for auto-saving data and searching
- New backends for Tektronix 5014C AWG, Alazar ADC, Qubic.
- Resonator fitting models for reflection/hanger geometries.
- Generic data processors for labeling/averaging results.

### Changed

- Upgraded supported python version to 3.12
- Renamed `SequenceElement` to `Timeline`.
- Renamed `ModulationFrequency` to `Frame` to align more closely with OpenQASM terminology.

