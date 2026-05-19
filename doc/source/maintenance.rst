==========================
Maintenance and operations
==========================

This page documents the conventions and procedures used to maintain, test, release, and
operate Neo. It is aimed at maintainers, release managers, packagers, and contributors
who want to test changes locally before submitting them.

Day-to-day development workflow is covered separately in :doc:`contributing`.


.. _section-versioning:

Versioning scheme
=================

Neo version numbers follow a semantic-versioning-compatible ``MAJOR.MINOR.PATCH`` pattern
(e.g. ``0.14.4``, ``0.15.0``). The in-development version on the ``master`` branch carries
a ``dev`` suffix (e.g. ``0.15.0dev``).

- **Patch** releases (``0.14.3`` → ``0.14.4``) contain bug fixes only. They do not change
  the public API.
- **Minor** releases (``0.14.x`` → ``0.15.0``) may add features and may deprecate parts of
  the public API. They aim to remain backwards-compatible with the previous minor release.
- **Major** releases (``0.x`` → ``1.0``, or ``1.x`` → ``2.x``) may remove deprecated APIs
  and contain breaking changes.

The canonical version string is set in :file:`pyproject.toml`.


.. _section-deprecation-policy:

Deprecation policy
==================

When changing or removing part of the public API, mark the old behaviour with a
``DeprecationWarning`` for at least one minor release before removing it. For example,
an API deprecated in ``0.15.0`` may be removed no earlier than ``0.16.0``.

Each deprecation should be mentioned in the release notes of both the release that
introduces the warning and the release that removes the API.


For testers
===========

The fastest way to verify a Neo change is to run the core test suite, which does not need
any external data::

    pytest neo/test/coretest

The IO and RawIO test suites operate on the
`ephy_testing_data <https://gin.g-node.org/NeuralEnsemble/ephy_testing_data>`_ dataset,
which is downloaded automatically via :mod:`datalad` on the first run. Installing
``datalad`` and ``git-annex`` is described in :doc:`contributing`.

If you need to run the test suite offline, or while packaging Neo for a Linux
distribution, set the ``NEO_TESTS_NO_NETWORK`` environment variable to skip tests that
require downloading data::

    NEO_TESTS_NO_NETWORK=True pytest

If a test fails on your machine but the corresponding CI job passes on ``master``, please
report it as a bug on the
`issue tracker <https://github.com/NeuralEnsemble/python-neo/issues>`_, including your
operating system, Python version, NumPy version, and a copy of the failing output.


For operators and maintainers
=============================

Building the documentation locally
----------------------------------

::

    pip install -e .[docs]
    cd doc
    make html

The built HTML is in :file:`doc/build/html/`. Open :file:`index.html` in a browser.


Read the Docs
-------------

The hosted documentation at https://neo.readthedocs.io is built automatically by Read the
Docs on every push to ``master`` and on every release tag. The build is configured by
:file:`.readthedocs.yaml`. New release tags appear automatically as selectable versions
once the build succeeds; if a new tag does not appear in the version switcher, activate it
manually on the `Read the Docs project page <https://readthedocs.org/projects/neo/>`_.


.. _section-release-process:

Making a release
----------------

The full release procedure:

1. Decide on the next version number following the :ref:`section-versioning` scheme.
2. Open a release-prep pull request that:

   - Updates the ``version`` field in :file:`pyproject.toml` (drop the ``dev`` suffix).
   - Adds a new release-notes file under :file:`doc/source/releases/` and links it from
     :file:`doc/source/releases.rst`.
   - Updates :file:`codemeta.json`: bump the ``version`` field, set ``dateModified`` to
     today's date, and set ``downloadUrl`` to ``null`` (the real download URL is not yet
     known — it is filled in after the PyPI upload, see step 8).
   - Updates any version-dependent references in the docs.

3. Merge the release-prep PR once CI is green.
4. Tag the merge commit on ``master``::

       git tag <version>
       git push --tags upstream

5. Wait for the core-test and io-test workflows to complete on the tagged commit. If any
   fail, fix them and re-tag (delete the broken tag first).
6. Trigger the **Release to PyPI** workflow manually from the GitHub Actions UI
   (``.github/workflows/publish-to-pypi.yml``). The workflow builds the sdist and wheel
   and uploads them to PyPI using the ``PYPI_API_TOKEN`` secret.
7. Verify that the release appears on `PyPI <https://pypi.org/project/neo/>`_ and that
   the new version is built and visible on
   `Read the Docs <https://readthedocs.org/projects/neo/>`_.
8. Copy the sdist download URL from the new release's PyPI page (``Download files`` →
   ``neo-X.Y.Z.tar.gz``) and update the ``downloadUrl`` field in :file:`codemeta.json` to
   that URL.
9. Open a follow-up PR that bumps the version in :file:`pyproject.toml` to the next
   ``X.Y.0dev`` and contains the :file:`codemeta.json` ``downloadUrl`` update from step 8.


EBRAINS spack package
---------------------

Neo is also distributed through the EBRAINS spack-builds repository as ``py-neo``:
https://gitlab.ebrains.eu/ri/tech-hub/platform/esd/ebrains-spack-builds/-/tree/master/packages/py-neo

After a PyPI release, open a merge request against ``ebrains-spack-builds`` to add the
new version and its SHA-256 checksum to the recipe.


Refreshing test data
--------------------

Test data lives in the separate ``ephy_testing_data`` repository on G-Node:
https://gin.g-node.org/NeuralEnsemble/ephy_testing_data

To add a new test file for a new or existing IO module, commit it to that repository (push
access is granted by the NeuralEnsemble maintainers). CI picks up the new commit
automatically because the cache key in :file:`.github/workflows/caches_cron_job.yml`
includes the upstream HEAD SHA.


Bumping dependencies and the supported Python version
-----------------------------------------------------

When raising the minimum supported version of Python, NumPy, or another dependency:

- Update the constraint in :file:`pyproject.toml` (both the relevant entry in
  ``dependencies`` and the ``Programming Language :: Python :: 3.X`` classifiers, plus
  ``requires-python``).
- Update the CI matrices in :file:`.github/workflows/core-test.yml` and
  :file:`.github/workflows/io-test.yml`.
- Update :doc:`install` and this page if any version numbers appear in the prose.


Technical debt
==============

Technical-debt items are tracked as GitHub issues with the ``technical-debt`` label.
Maintainers review the backlog periodically and prioritise items alongside feature work
when planning the next release.
