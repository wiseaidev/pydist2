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

.. image:: https://img.shields.io/pypi/status/pydist2.svg
        :target: https://pypi.python.org/pypi/pydist2/

.. image:: https://img.shields.io/pypi/wheel/pydist2.svg
        :target: https://pypi.python.org/pypi/pydist2/

.. image:: https://img.shields.io/github/license/Harmouch101/pydist2.svg
        :target: https://github.com/Harmouch101/pydist2


pydist2 is a python library that provides a set of methods for calculating distances between observations.
There are two main classes:

* **pdist1** which calculates the pairwise distances between observations in one matrix and returns a distance matrix.
* **pdist2** computes the distances between observations in two matrices and also returns a distance matrix.

Usage
-----
.. code-block:: console

   pdist1(P, metric = "euclidean", matrix=False)
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

Progress & Features
-------------------

- [X] Commit the first code's version.
- [X] Support the following `list of distances`_. 
- [X] Display the distance in a matrix form(a combination for each pair of points)::

   >>> X = np.array([[100, 100],[0, 100],[100, 0], [500, 400], [300, 600]])
   >>> pdist1(X,matrix=True) # by default, metric = 'euclidean'
   array([[100.    , 100.    , 100.    ,   0.    , 100.    ],
          [100.    , 100.    , 100.    , 100.    ,   0.    ],
          [500.    , 100.    , 100.    , 500.    , 400.    ],
          [538.5165, 100.    , 100.    , 300.    , 600.    ],
          [141.4214,   0.    , 100.    , 100.    ,   0.    ],
          [583.0952,   0.    , 100.    , 500.    , 400.    ],
          [583.0952,   0.    , 100.    , 300.    , 600.    ],
          [565.6854, 100.    ,   0.    , 500.    , 400.    ],
          [632.4555, 100.    ,   0.    , 300.    , 600.    ],
          [282.8427, 500.    , 400.    , 300.    , 600.    ]])

where the first column represents the distance between each pair of observations. for instance, the euclidean distance between (100. , 100.) and ( 0. , 100.) is 100.

- [X] Support numpy arrays of the same size only.

Todo list
---------

- [ ] Re-validate the correctness of the distances equations.
- [ ] Performance tests & vectorization.
- [ ] Adding new distances.
- [ ] Adding a squared form of the distance.
- [ ] Support tuples and list.
- [ ] Write more test cases.
- [ ] Handling Exceptions.
- [ ] Restructure the docs.

.. _MIT License: https://opensource.org/licenses/MIT
.. _Pypi: https://pypi.org/project/pydist2/
.. _readthedocs: https://pydist2.readthedocs.io
.. _list of distances: https://pydist2.readthedocs.io/en/latest/guide.html
