.. _install:

Installation
============

Prerequisites
-------------

- h-NNE requires python >= 3.7

It should be straightforward to install the package and dependencies with pip. Below are comments on some dependencies which the user might want to be aware of:

- Currently the numpy version is fixed to 1.20, hopefully we will be able to remove this restriction if coming versions.
- h-NNE depends on pynndescent which in turn requires numba. Make sure those packages properly function in your system, otherwise you might experience performance issues on the approximate nearest neighbor step of h-nne.


Installation
------------

To install h-NNE with pip, execute:

.. code-block:: bash

    pip install hnne