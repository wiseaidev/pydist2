=======
pydist2
=======


.. image:: https://img.shields.io/pypi/v/pydist2.svg
        :target: https://pypi.python.org/pypi/pydist2

.. image:: https://img.shields.io/travis/Harmouch101/pydist2.svg
        :target: https://travis-ci.com/Harmouch101/pydist2

.. image:: https://readthedocs.org/projects/pydist2/badge/?version=latest
        :target: https://pydist2.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status


pydist2 is a python library that provides a set of methods for calculating distances between observations.
There are two main classes:

* **pdist1** which calculates the pairwise distances between observations in one matrix and returns a distance matrix.
* **pdist2** computes the distances between observations in two matrices and also returns a distance matrix.

Usage
-----
.. code-block:: console

   pdist1(P, metric = "euclidean",)
   pdist2(P, Q, metric = "minkowski", exp = 3)

**Arguments**: 

* two matrices P and Q.
* metric: The distance function to use.
* exp: The exponent of the Minkowski distance.

Installation
-------------

The pydist2 library is available on Pypi_. Thus, you can install the latest available version using *pip*::

   $pip install pydist2

Supported Python versions
-------------------------

pydist2 has been tested with Python 3.7 and 3.8. 

For more information, please checkout the documentation which is available at readthedocs_.

This program and the accompanying materials are made available under the terms of the `MIT License`_.

.. _MIT License: https://opensource.org/licenses/MIT
.. _Pypi: https://pypi.org/project/pydist2/
.. _readthedocs: https://pydist2.readthedocs.io
