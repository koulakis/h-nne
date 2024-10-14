.. _contributions_guide:

Contributions
=============

Be it a small correction in the documentation or a major new feature, all contributions are welcome. Please, make sure
to contact us before working on a contribution so we can agree on the scope and the changes.

General guidelines
------------------

Please make sure:

- inform us github before starting work on a contribution
- your code is clean and efficient
- you use python typing
- you test any new functionality


Codebase setup
--------------

Install the project locally it locally with:

.. code-block:: python

    pip install -e ".[test]"

This will install the project together with the requirements to run testing.

Linting
-------

We use pre-commit to keep a consistent code style. To run it:

.. code-block:: bash

    pre-commit run --all-files

This will automatically format the code, except from the flake8 issues which need to be manually resolved. Once fixed,
add again your changes with git.

Testing
-------

To run all tests:

.. code-block:: python

    DATA_PATH_HNNE=<path to test datasets> pytest

Note the `DATA_PATH_HNNE` environment variable. This is a location where some torchvision datasets will be stored to be
used in some of the tests. Those are currently the test parts of MNIST, FMNIST and CIFAR10 which occupy approximately
200 MB of disk space.

Upload new version to PyPi (for admins)
---------------------------------------

To upload a version you need to have `twine` installed and access to the PiPy project: https://pypi.org/project/hnne.
To upload a new version:

- make sure you updated the version of the package in `setup.py` to the new one
- ensure the code is merged in the `main` branch
- create and push a new tag with the version number, e.g. for version 0.1.5:
.. code-block:: bash

    git tag 0.1.5.
    git push origin --tags
- Create a source distribution:
.. code-block:: bash

    python setup.py sdist
- Upload the distribution with twine (you will be asked for your PiPy credentials):
.. code-block:: bash

    twine upload dist/*

- Check that the new version is in https://pypi.org/project/hnne
- Build a new documentation version in https://hnne.readthedocs.io/en/latest
