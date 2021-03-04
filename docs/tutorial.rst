====================
pydist2 tutorial
====================

This module contains two main interfaces which provides the following
functionalities:

* **pdist1** calculates the pairwise distances between points in one vector and returns a vector which represents the distance between the observations.

* **pdist2** calculates the distances between points in two vectors and returns a vector which represents the distance between the observations.

The beginner will enjoy how the :mod:`~pydist2.distance` module lets you get
started quickly.

>>> import numpy as np
>>> from pydist2.distance import Euclidean
>>> x = np.array([[1, 2, 3],
       [7, 8, 9],
       [5, 6, 7],], dtype=np.float32)
>>> Euclidean.compute(x)
array([10.39230485,  6.92820323,  3.46410162])

However, a better programmer can use the :class:`~pydist2.distance.pdist1`
or :class:`~pydist2.distance.pdist2` class to compute the actual pairwise
vectors distances *object* upon which he can then perform lots of operations.
For example, consider this Python program:

.. code-block:: python3
   
  import numpy as np
  from pydist2.distance import pdist1
  x = np.array([[1, 2, 3],
      [7, 8, 9],
      [5, 6, 7],], dtype=np.float32)
  euclidean_distance = pdist1(x,'euclidean')
  print(f"The euclidean distance between each observation in \n{x}\n is:\n{euclidean_distance}")

this program will produce the following output:

.. code-block:: console
  
   The euclidean distance between each observation in 
   [[1. 2. 3.]
    [7. 8. 9.]
    [5. 6. 7.]]
    is:
   [10.39230485  6.92820323  3.46410162]

Read :doc:`guide` to learn more!

.. warning::

   This module only handles numpy arrays as inputs;
   any other data types are right out(e.g. list, tuple...);
   this will be solved in future develpment of the package.

