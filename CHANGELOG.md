# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.1] - 2026
### Added
- Fixed atoms functionality
- Ability to directly input stepsize
### Fixed
- Minor bug fixes and stability improvements

## [1.0.0] - 2025

### Added
- Initial public release of ML-FSM
- Internal coordinates interpolation for the Freezing String Method
- Support for ML-based potentials via ASE calculator interface
- Support for AIMNet2, MACEOFF23, FAIR UMA, TensorNet, xTB, and QChem backends
- Google Colab example notebook for Diels-Alder reaction with AIMNet2
- Comprehensive example script (`examples/fsm_example.py`) covering most FSM functionality
- Custom ASE calculator wrapper templates for NNPs without native ASE interfaces
- Full test suite with pytest
- Type annotations and mypy support
- Sphinx documentation hosted on Read the Docs
- Pixi-based development environment

[Unreleased]: https://github.com/thegomeslab/ML-FSM/compare/v1.0.1...HEAD
[1.0.1]: https://github.com/thegomeslab/ML-FSM/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/thegomeslab/ML-FSM/releases/tag/v1.0.0
