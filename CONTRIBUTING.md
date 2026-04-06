# Contributing to ML-FSM

Thank you for your interest in contributing to ML-FSM! This document outlines the process for contributing and how to set up your development environment.

## Project Maintainers

ML-FSM is primarily developed and maintained at the [Gomes Lab](https://github.com/thegomeslab) at the University of Iowa. The package was co-created by Jonah Marks (now at AstraZeneca), who remains a maintainer, with additional contributions from Jonathon Vandezande.

For questions, bug reports, or to discuss potential contributions, please contact:

**Joe Gomes** — joe-gomes@uiowa.edu

## Before You Open a Pull Request

**Please open an issue or reach out before starting significant work.**

If you have a bug report, feature request, or want to propose a change, start by [opening an issue](https://github.com/thegomeslab/ML-FSM/issues) on GitHub. For questions or to discuss a contribution before diving in, you can also contact Joe directly at the address above.

This helps avoid duplicated effort and ensures your contribution aligns with the project's direction before you invest time in it.

## Development Setup

ML-FSM uses [Pixi](https://pixi.sh) to manage the development environment. Pixi handles dependencies, virtual environments, and task running — no manual conda/pip setup needed.

### 1. Install Pixi

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

Or see the [Pixi installation docs](https://pixi.sh/latest/#installation) for other options.

### 2. Clone the repository

```bash
git clone https://github.com/thegomeslab/ML-FSM.git
cd ML-FSM
```

### 3. Set up the dev environment

```bash
pixi install -e dev
```

This creates an isolated environment with all development dependencies (pytest, ruff, mypy, sphinx, pre-commit, etc.) — no additional steps required.

### 4. Install pre-commit hooks

```bash
pixi run -e dev pre-commit install
```

This registers the hooks so they run automatically on every commit.

## Available Commands

All development tasks are run via `pixi run` from the repository root. Make sure to use the `dev` environment (`-e dev`) or set it as your active environment.

| Command | Description |
|---|---|
| `pixi run fmt` | Format code with ruff |
| `pixi run lint` | Lint and auto-fix code with ruff |
| `pixi run types` | Run static type checking with mypy |
| `pixi run test` | Run the test suite with pytest |
| `pixi run coverage` | Run tests with coverage report (generates `coverage.xml`) |
| `pixi run docs` | Build the Sphinx documentation locally |
| `pixi run all` | Run fmt, lint, types, test, and docs in sequence |

For example:

```bash
pixi run test          # run tests
pixi run fmt && pixi run lint  # format then lint
pixi run all           # run everything
```

## Pre-commit Checks

The pre-commit hooks run ruff format, ruff check, mypy, and pytest on every commit and push. Before opening a pull request, run the full pre-commit suite manually to make sure everything passes:

```bash
pixi run -e dev pre-commit run --all-files
```

Pull requests that fail any of these checks will not be merged.

## Contribution Workflow

1. [Open an issue](https://github.com/thegomeslab/ML-FSM/issues) or contact Joe to discuss your proposed change.
2. Fork the repository and create a feature branch from `main`.
3. Make your changes, adding tests for any new behavior.
4. Run `pixi run all` to confirm everything passes.
5. Run `pre-commit run --all-files` as a final check.
6. Open a pull request against `main`, referencing the relevant issue.

## Code Style

This project uses [ruff](https://github.com/astral-sh/ruff) for formatting and linting, and [mypy](https://mypy-lang.org/) for static type checking. Docstrings follow the NumPy convention. The pre-commit hooks enforce these automatically — just run `pixi run fmt` and `pixi run lint` before committing.
