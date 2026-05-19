# Contributing to Neo

Contributions of all kinds to Neo are welcome. This includes bug reports, bug fixes, new file formats, documentation improvements, and new features.

This file gives an overview of how to contribute, with pointers to more extensive guidance in the documentation.

## Getting started

- **How to report a bug or request a feature:** see [Bug reports](https://neo.readthedocs.io/en/latest/bug_reports.html). Issues are tracked at <https://github.com/NeuralEnsemble/python-neo/issues>.
- **Setting up a development environment, branching, testing, and submitting a pull request:** see the [Contributing guide](https://neo.readthedocs.io/en/latest/contributing.html).
- **Adding support for a new file format:** see [Adding a new IO module](https://neo.readthedocs.io/en/latest/add_file_format.html).
- **How the project is run (maintainers, decision-making, becoming a maintainer):** see [Governance](https://neo.readthedocs.io/en/latest/governance.html).
- **Versioning scheme, deprecation policy, releases, refreshing test data, updating the EBRAINS spack recipe:** see [Maintenance and operations](https://neo.readthedocs.io/en/latest/maintenance.html).

## Communication

Discussion of new features and design questions takes place on the NeuralEnsemble mailing list; discussion of specific bugs or PRs takes place on the relevant GitHub issue or pull request. Channels are listed in the [Contributing guide](https://neo.readthedocs.io/en/latest/contributing.html).

## Code style and conventions

- Code is formatted with [Black](https://black.readthedocs.io) using a line length of 120 (configured in `pyproject.toml`). A weekly CI workflow opens a PR if formatting drifts.
- Docstrings use the [NumPy docstring style](https://numpydoc.readthedocs.io/en/latest/format.html).
- Public API changes should be accompanied by tests under `neo/test/` and, where user-visible, an entry in the relevant `doc/source/releases/*.rst` release notes file.

## Code of Conduct

By participating in this project you agree to abide by the [NumFOCUS Code of Conduct](CODE_OF_CONDUCT.md).

## Licensing of contributions

Neo is released under the [BSD-3-Clause license](LICENSE.txt). By submitting a pull request, you agree that your contribution may be distributed under the same license.
