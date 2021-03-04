#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
| The following script implements some useful vectors distances calculation.

| The code has been translated from matlab and follows the same logic.
| The original `matlab code`_ belongs to `Piotr Dollar's Toolbox`_.
| Please refer to the above web pages for more explanations.
| Matlab code can also be found @ MathWorks_.
| This program and the accompanying materials are made available under the terms of the `MIT License`_.
| SPDX short identifier: MIT
| Contributors:
    Mahmoud Harmouch, mail_.

.. _matlab code: https://github.com/pdollar/toolbox/blob/master/classify/pdist2.m
.. _Piotr Dollar's Toolbox: http://vision.ucsd.edu/~pdollar/toolbox/doc/index.html
.. _MathWorks: https://www.mathworks.com/matlabcentral/mlc-downloads/downloads/submissions/29004/versions/2/previews/FPS_in_image/FPS%20in%20image/Help%20Functions/SearchingMatches/pdist2.m/index.html
.. _MIT License: https://opensource.org/licenses/MIT
.. _mail: mahmoudddharmouchhh@gmail.com
"""

from abc import ABC, abstractmethod

import numpy as np

from .custom_typing import NumpyArray, PositiveInteger, String, Void


class VectorsDistanceDescriptor(ABC):
    """
    This descriptor constuct a blueprint for different vectors distance.

    It defines the methods specified in the subclasses. All of the subclasses
    attributes(fields) are prefixed with an underscore, since they are not
    intended to be accessed directly, but rather through the getter and setter
    methods. They are unlikely to change, but must be defiened in a distance
    subclass in order to be compatible with the `distance` module. Any custom
    methods in a subclass should not be prefixed with an underscore. All of
    these methods must be implemented in any subclass in order to work with.
    Any implementation specific logic should be handled in a subclass.
    """

    @abstractmethod
    def __init__(self, metric: String) -> Void:
        """
        Return an instance of the Distance class based on a metric.

        :param metric: String that represents the name of the distance method.
        :return: Void.
        """
        pass

    @property
    @abstractmethod
    def metric(self) -> String:
        """
        A getter method that returns the metric attribute.

        :param: Instance of the class.
        :return: String that represents the metric attribute.
        """
        pass

    @metric.setter
    @abstractmethod
    def metric(self, value: String) -> Void:
        """
        A setter method that changes the value of the metric attribute.

        :param value: String that specifies the name of the distance method.
        :return: Void.
        """
        pass

    @classmethod
    @abstractmethod
    def compute(self, P: NumpyArray, Q: NumpyArray) -> NumpyArray:
        """
        A method that computes the distance between two vectors P and Q.

        :param P: NumpyArray that represents the first vector/matrix.
        :param Q: NumpyArray that represents the second vector/matrix.
        :return: The distance between the two vectors.
        """
        pass

    @abstractmethod
    def __repr__(self) -> String:
        """
        A method that Return a formated string for a given instance.

        :param: a reference for a given instance.
        :return: a formated string for a given instance.
        """
        pass


class PairwiseDistanceDescriptor(ABC):
    """
    This descriptor construct a blueprint for different pairwise distances.

    It defines the methods specified in the subclasses. All of the subclasses
    attributes(fields) are prefixed with an underscore, since they are not
    intended to be accessed directly, but rather through the getter and setter
    methods. They are unlikely to change, but must be defiened in a distance
    subclass in order to be compatible with the `distance` module. Any custom
    methods in a subclass should not be prefixed with an underscore. All of
    these methods must be implemented in any subclass in order to work with.
    Any implementation specific logic should be handled in a subclass.
    """

    @abstractmethod
    def __init__(self, metric: String) -> Void:
        """
        Return an instance of the Distance class based on the metric.

        :param metric: a given name of the distance method.
        :return: void.
        """
        pass

    @property
    @abstractmethod
    def metric(self) -> String:
        """
        A getter method that returns the metric attribute.

        :param: Instance of the class.
        :return: metric.
        """
        pass

    @metric.setter
    @abstractmethod
    def metric(self, value: String) -> Void:
        """
        A setter method that changes the value of the metric attribute.

        :param value: String that specifies the name of the distance method.
        :return: Void.
        """
        pass

    @classmethod
    @abstractmethod
    def compute(self, P: NumpyArray) -> NumpyArray:
        """
        A method that computes the pairwise distance of vector P.

        :param P: NumpyArray that represents a certain vector.
        :return: NumpyArray that contains the 'metric' distance between
            each pair of data for the vector P.
        """
        pass

    @abstractmethod
    def __repr__(self) -> String:
        """
        A method that Return a formated string for a given instance.

        :param self: a reference for a given instance.
        :return: a formated string for a given instance.
        """
        pass


class L1Distance(VectorsDistanceDescriptor):
    """
    The L1 norm distance between two vectors. Also known as Manhattan Distance.

    Literature:
        https://en.wikipedia.org/wiki/Taxicab_geometry
    """

    _metric = String("metric")

    def __init__(self, metric: String = "L1 Distance") -> Void:
        """Returns an instance of a L1Distance class."""
        self._metric = metric

    @property
    def metric(self) -> String:
        """
        A getter method that returns the metric attribute.

        :param: instance of the class.
        :return: metric.
        """
        return self._metric

    @metric.setter
    def metric(self, value: String) -> Void:
        """
        A setter method that changes the value of the metric attribute.

        :param value: a string that specifies the name of the distance method.
        :return: Void.
        """
        self._metric = value

    @classmethod
    def compute(self, P: NumpyArray, Q: NumpyArray) -> NumpyArray:
        """Compute the distance using sum(abs(P-Qi))."""
        m = P.shape[0]
        n = Q.shape[0]
        mOnes = np.ones((1, m), dtype=np.float64)
        D = np.zeros((m, n), dtype=np.float64)
        for i in range(n):
            Qi = Q[i, :]
            Qi = Qi * mOnes
            D[:, i] = np.sum(np.abs(P - Qi), axis=1)
        return D

    def __repr__(self) -> String:
        """Custom repr function."""
        return f"{self.__dict__}"


class SquaredEuclideanDistance(VectorsDistanceDescriptor):
    """The L2 norm distance squared.

    Literature:
        https://en.wikipedia.org/wiki/Euclidean_distance#Squared_Euclidean_distance
    """

    _metric = String("metric")

    def __init__(self, metric: String = "Squared Euclidean Distance") -> Void:
        """Return an instance of a SquaredEuclideanDistance class."""
        self._metric = metric

    @property
    def metric(self) -> String:
        """
        A getter method that returns the metric attribute.

        :param: instance of the class.
        :return: metric.
        """
        return self._metric

    @metric.setter
    def metric(self, value: String) -> Void:
        """
        A setter method that changes the value of the metric attribute.

        :param value: a string that specifies the name of the distance method.
        :return: Void.
        """
        self._metric = value

    @classmethod
    def compute(self, P: NumpyArray, Q: NumpyArray) -> NumpyArray:
        """Compute the Squared Euclidean distance using (P - Q)^2."""
        m = P.shape[0]
        n = Q.shape[1]
        mOnes = np.ones((m, n), dtype=np.float64)
        Qt = Q.transpose()
        PP = (np.sum(P * P, axis=1) * mOnes).transpose()
        QQ = np.sum(Qt * Qt, axis=0) * mOnes
        return PP + QQ - 2 * np.dot(P, Qt)

    def __repr__(self) -> String:
        """Custom repr function."""
        return f"{self.__dict__}"


class EuclideanDistance(SquaredEuclideanDistance):
    """The L2 norm distance, aka the euclidean Distance."""

    def __init__(self, metric: String = "Euclidean Distance") -> Void:
        """Return an instance of a EuclideanDistance class."""
        super().__init__(metric)

    @classmethod
    def compute(self, P: NumpyArray, Q: NumpyArray) -> NumpyArray:
        """Compute the distance using sqrt(Squared Euclidean Distance)."""
        return np.sqrt(super().compute(P, Q))


class ChiSquaredDistance(VectorsDistanceDescriptor):
    """
    Compute the Chi-Squared Distance using the formula below.

    d(P,Q) = sum((Pi - Qi)^2 / (Pi + Qi)) / 2

    Literature:
        https://www.hindawi.com/journals/mpe/2015/352849/
    """

    _metric = String("metric")

    def __init__(self, metric: String = "Chi-Squared Distance") -> Void:
        """Return an instance of a Chi-Squared class."""
        self._metric = metric

    @property
    def metric(self) -> String:
        """
        A getter method that returns the metric attribute.

        :param: instance of the class.
        :return: metric.
        """
        return self._metric

    @metric.setter
    def metric(self, value: String) -> Void:
        """
        A setter method that changes the value of the metric attribute.

        :param value: a string that specifies the name of the distance method.
        :return: Void.
        """
        self._metric = value

    @classmethod
    def compute(self, P: NumpyArray, Q: NumpyArray) -> NumpyArray:
        """Compute the Chi-Squared Distance."""
        m = P.shape[0]
        n = Q.shape[0]
        mOnes = np.ones((1, m), dtype=np.float64)
        D = np.zeros((m, n), dtype=np.float64)
        eps = np.finfo(float).eps
        for i in range(n):
            Qi = Q[i, :]
            QiRep = Qi * mOnes
            d = QiRep - P
            s = QiRep + P
            D[:, i] = np.sum((d * d) / (s + eps), axis=1)
        return D / 2

    def __repr__(self) -> String:
        """Custom repr function."""
        return f"{self.__dict__}"


class CosineDistance(VectorsDistanceDescriptor):
    """
    Compute the Cosine Distance equals tp 1 - Cosine_Similarity.

    Literature:
        https://en.wikipedia.org/wiki/Cosine_similarity
    """

    _metric = String("metric")

    def __init__(self, metric: String = "Cosine Distance") -> Void:
        """Return an instance of a CosineDistance class."""
        self._metric = metric

    @property
    def metric(self) -> String:
        """
        A getter method that returns the metric attribute.

        :param: instance of the class.
        :return: metric.
        """
        return self._metric

    @metric.setter
    def metric(self, value: String) -> Void:
        """
        A setter method that changes the value of the metric attribute.

        :param value: a string that specifies the name of the distance method.
        :return: Void.
        """
        self._metric = value

    @classmethod
    def compute(self, P: NumpyArray, Q: NumpyArray) -> NumpyArray:
        """Distance is defined as 1 - cosine(angle between two vectors)."""
        p = P.shape[1]
        pOnes = np.ones((1, p), dtype=np.float64)
        PP = (pOnes.T * np.sqrt(np.sum(P * P, axis=1))).T
        QQ = (pOnes.T * np.sqrt(np.sum(Q * Q, axis=1))).T
        P = P.astype(np.float64)
        Q = Q.astype(np.float64)
        np.divide(P, PP, out=P, where=PP != 0)
        np.divide(Q, QQ, out=Q, where=QQ != 0)

        return 1 - np.dot(P, Q.T)

    def __repr__(self) -> String:
        """Custom repr function."""
        return f"{self.__dict__}"


class EarthMoversDistance(VectorsDistanceDescriptor):
    """
    Compute the Earth Mover's Distance between two vectors.

    Literature:
        https://en.wikipedia.org/wiki/Earth_mover%27s_distance
    """

    _metric = String("metric")

    def __init__(self, metric: String = "Earth Mover's Distance") -> Void:
        """Return an instance of a EarthMoversDistance class."""
        self._metric = metric

    @property
    def metric(self) -> String:
        """
        A getter method that returns the metric attribute.

        :param: instance of the class.
        :return: metric.
        """
        return self._metric

    @metric.setter
    def metric(self, value: String) -> Void:
        """
        A setter method that changes the value of the metric attribute.

        :param value: a string that specifies the name of the distance method.
        :return: Void.
        """
        self._metric = value

    @classmethod
    def compute(self, P: NumpyArray, Q: NumpyArray) -> NumpyArray:
        """Distance is defined as cosine of the angle between two vectors."""
        m = P.shape[1]
        n = Q.shape[1]
        Pcdf = np.cumsum(P, axis=1)
        Qcdf = np.cumsum(Q, axis=1)
        mOnes = np.ones((1, m), dtype=np.float64)
        D = np.zeros((m, n), dtype=np.float64)
        for i in range(n):
            qcdf = Qcdf[i, :]
            qcdfRep = qcdf * mOnes
            D[:, i] = np.sum(np.abs(Pcdf - qcdfRep), axis=1)
        return D

    def __repr__(self) -> String:
        """Custom repr function."""
        return f"{self.__dict__}"


class Euclidean(PairwiseDistanceDescriptor):
    """Pairwise Euclidean distance."""

    _metric = String("metric")

    def __init__(self, metric: String = "Pairwise Euclidean Distance") -> Void:
        """Return an instance of a Euclidean class."""
        self._metric = metric

    @property
    def metric(self) -> String:
        """
        A getter method that returns the metric attribute.

        :param: instance of the class.
        :return: metric.
        """
        return self._metric

    @metric.setter
    def metric(self, value: String) -> Void:
        """
        A setter method that changes the value of the metric attribute.

        :param value: a string that specifies the name of the distance method.
        :return: Void.
        """
        self._metric = value

    @classmethod
    def compute(self, P: NumpyArray) -> NumpyArray:
        """Compute the pairwise distances between each elements of P."""
        m, n = P.shape[: 2]
        D = np.zeros((1, int(m * (m - 1) / 2)), dtype=np.float64)
        k = 0
        for i in range(m):
            dsq = np.zeros((m - i - 1, 1), dtype=np.float64)
            for q in range(n):
                dsq = dsq + (P[i, q] - P[i + 1: m, q]).reshape(-1, 1) ** 2
            D[0, k: k + m - i - 1] = np.sqrt(dsq).reshape(1, -1)
            k += m - i - 1
        return D[0]

    def __repr__(self) -> String:
        """Custom repr function."""
        return f"{self.__dict__}"


class StandardizedEuclidean(PairwiseDistanceDescriptor):
    """Pairwise Standardized Euclidean Distance,Weighted Euclidean distance."""

    _metric = String("metric")

    def __init__(self, metric: String = "Pairwise Standardized Euclidean Distance") -> Void:
        """Return an instance of a StandardizedEuclidean class."""
        self._metric = metric

    @property
    def metric(self) -> String:
        """
        A getter method that returns the metric attribute.

        :param: instance of the class.
        :return: metric.
        """
        return self._metric

    @metric.setter
    def metric(self, value: String) -> Void:
        """
        A setter method that changes the value of the metric attribute.

        :param value: a string that specifies the name of the distance method.
        :return: Void.
        """
        self._metric = value

    @classmethod
    def compute(self, P: NumpyArray) -> NumpyArray:
        """Compute the Weighted Euclidean distance as: sqrt(wgts*(P_iq-P_hq)^2)."""
        m, n = P.shape[: 2]
        wgts = 1 / np.var(P, ddof=1, axis=0)
        D = np.zeros((1, int(m * (m - 1) / 2)), dtype=np.float64)
        k = 0
        for i in range(m):
            dsq = np.zeros((m - i - 1, 1), dtype=np.float64)
            for q in range(n):
                dsq = dsq + wgts[q] * (P[i, q] - P[i + 1: m, q]).reshape(-1, 1) ** 2
            D[0, k: k + m - i - 1] = np.sqrt(dsq).reshape(1, -1)
            k += m - i - 1
        return D[0]

    def __repr__(self) -> String:
        """Custom repr function."""
        return f"{self.__dict__}"


class CityBlock(PairwiseDistanceDescriptor):
    """Pairwise City Block Distance."""

    _metric = String("metric")

    def __init__(self, metric: String = "City Block Distance") -> Void:
        """Return an instance of a CityBlock class."""
        self._metric = metric

    @property
    def metric(self) -> String:
        """
        A getter method that returns the metric attribute.

        :param: instance of the class.
        :return: metric.
        """
        return self._metric

    @metric.setter
    def metric(self, value: String) -> Void:
        """
        A setter method that changes the value of the metric attribute.

        :param value: a string that specifies the name of the distance method.
        :return: Void.
        """
        self._metric = value

    @classmethod
    def compute(self, P: NumpyArray) -> NumpyArray:
        """Compute the City Block distance: sum|P_iq-P_hq|."""
        m, n = P.shape[: 2]
        D = np.zeros((1, int(m * (m - 1) / 2)), dtype=np.float64)
        k = 0
        for i in range(m):
            dsq = np.zeros((m - i - 1, 1), dtype=np.float64)
            for q in range(n):
                dsq = dsq + np.abs((P[i, q] - P[i + 1: m, q]).reshape(-1, 1))
            D[0, k: k + m - i - 1] = dsq.reshape(1, -1)
            k += m - i - 1
        return D[0]

    def __repr__(self) -> String:
        """Custom repr function."""
        return f"{self.__dict__}"


class Mahalanobis(PairwiseDistanceDescriptor):
    """Pairwise Mahalanobis Distance."""

    _metric = String("metric")

    def __init__(self, metric: String = "Mahalanobis Distance") -> Void:
        """Return an instance of a Mahalanobis class."""
        self._metric = metric

    @property
    def metric(self) -> String:
        """
        A getter method that returns the metric attribute.

        :param: instance of the class.
        :return: metric.
        """
        return self._metric

    @metric.setter
    def metric(self, value: String) -> Void:
        """
        A setter method that changes the value of the metric attribute.

        :param value: a string that specifies the name of the distance method.
        :return: Void.
        """
        self._metric = value

    @staticmethod
    def cov(P0):
        """A helper method that computes the covariance for a given matrix."""
        P = P0 - P0.mean(axis=0)
        U, s, V = np.linalg.svd(P, full_matrices=0)
        D = np.dot(np.dot(V.T, np.diag(s**2)), V)
        return D / (P0.shape[0] - 1)

    @classmethod
    def compute(self, P: NumpyArray) -> NumpyArray:
        """Compute the Mahalanobis distance."""
        m, n = P.shape[: 2]
        D = np.zeros((1, int(m * (m - 1) / 2)), dtype=np.float64)
        # left division of cov(P) with np.eye to get the inverse matrix.
        # the reason to do so is because cov(P) is a singular matrix.
        # so np.linalg.inv(cov(P)) would raize numpy.linalg.LinAlgError.
        x, _, _, _ = np.linalg.lstsq(Mahalanobis.cov(P), np.eye(3), rcond=None)
        k = 0
        for i in range(m):
            del_ = P[i, :] - P[i + 1: m, :]
            dsq = np.sum(np.dot(del_, x) * del_, axis=1)
            D[0, k: k + m - i - 1] = np.sqrt(dsq).reshape(1, -1)
            k += m - i - 1
        return D[0]

    def __repr__(self) -> String:
        """Custom repr function."""
        return f"{self.__dict__}"


class Minkowski(PairwiseDistanceDescriptor):
    """Pairwise Minkowski Distance."""

    _metric = String("metric")

    def __init__(self, metric: String = "Minkowski Distance") -> Void:
        """Return an instance of a ManhattanDistance class."""
        self._metric = metric

    @property
    def metric(self) -> String:
        """
        A getter method that returns the metric attribute.

        :param: instance of the class.
        :return: metric.
        """
        return self._metric

    @metric.setter
    def metric(self, value: String) -> Void:
        """
        A setter method that changes the value of the metric attribute.

        :param value: a string that specifies the name of the distance method.
        :return: Void.
        """
        self._metric = value

    @classmethod
    def compute(self, P: NumpyArray, exp: PositiveInteger = 3) -> NumpyArray:
        """
        Compute the City Block distance: (sum(P_iq-P_hq)^(exp))^(1/exp).

        exp = 2 ----> Euclidean distance
        exp = 1 ----> city-block distance
        """
        m, n = P.shape[: 2]
        D = np.zeros((1, int(m * (m - 1) / 2)), dtype=np.float64)
        k = 0
        for i in range(m):
            dpow = np.zeros((m - i - 1, 1), dtype=np.float64)
            for q in range(n):
                dpow = dpow + np.abs((P[i, q] - P[i + 1: m, q]).reshape(-1, 1)) ** exp
            D[0, k: k + m - i - 1] = dpow.reshape(1, -1) ** (1 / exp)
            k += m - i - 1
        return D[0]

    def __repr__(self) -> String:
        """Custom repr function."""
        return f"{self.__dict__}"


class Chebychev(PairwiseDistanceDescriptor):
    """Pairwise Chebychev Distance."""

    _metric = String("metric")

    def __init__(self, metric: String = "Chebychev Distance") -> Void:
        """Return an instance of a Chebychev class."""
        self._metric = metric

    @property
    def metric(self) -> String:
        """
        A getter method that returns the metric attribute.

        :param: instance of the class.
        :return: metric.
        """
        return self._metric

    @metric.setter
    def metric(self, value: String) -> Void:
        """
        A setter method that changes the value of the metric attribute.

        :param value: a string that specifies the name of the distance method.
        :return: Void.
        """
        self._metric = value

    @classmethod
    def compute(self, P: NumpyArray) -> NumpyArray:
        """Compute the Chebychev distance."""
        m, n = P.shape[: 2]
        D = np.zeros((1, int(m * (m - 1) / 2)), dtype=np.float64)
        k = 0
        for i in range(m):
            dmax = np.zeros((m - i - 1, 1), dtype=np.float64)
            for q in range(n):
                dmax = np.maximum(dmax, np.abs(P[i, q] - P[i + 1: m, q]).reshape(-1, 1))
            D[0, k: k + m - i - 1] = dmax.reshape(-1)
            k += m - i - 1
        return D[0]

    def __repr__(self) -> String:
        """Custom repr function."""
        return f"{self.__dict__}"


class Cosine(PairwiseDistanceDescriptor):
    """Pairwise Cosine Distance."""

    _metric = String("metric")

    def __init__(self, metric: String = "Cosine Distance") -> Void:
        """Return an instance of a Cosine class."""
        self._metric = metric

    @property
    def metric(self) -> String:
        """
        A getter method that returns the metric attribute.

        :param: instance of the class.
        :return: metric.
        """
        return self._metric

    @metric.setter
    def metric(self, value: String) -> Void:
        """
        A setter method that changes the value of the metric attribute.

        :param value: a string that specifies the name of the distance method.
        :return: Void.
        """
        self._metric = value

    @classmethod
    def compute(self, P: NumpyArray) -> NumpyArray:
        """Compute the Cosine distance."""
        m, n = P.shape[: 2]
        mOnes = np.ones((m, n), dtype=np.float64)
        Pnorm = np.sqrt(np.sum(P ** 2, axis=1)).reshape(-1, 1) * mOnes
        P = P / Pnorm
        D = np.zeros((1, int(m * (m - 1) / 2)), dtype=np.float64)
        k = 0
        for i in range(m):
            d = np.zeros((m - i - 1, 1), dtype=np.float64)
            for q in range(n):
                d = d + (P[i, q] * P[i + 1: m, q]).reshape(-1, 1)
            d = np.where(d > 1, 1, d)
            D[0, k: k + m - i - 1] = (1 - d).reshape(-1)
            k += m - i - 1
        return D[0]

    def __repr__(self) -> String:
        """Custom repr function."""
        return f"{self.__dict__}"


class Correlation(PairwiseDistanceDescriptor):
    """Pairwise Correlation Distance."""

    _metric = String("metric")

    def __init__(self, metric: String = "Correlation Distance") -> Void:
        """Return an instance of a Correlation class."""
        self._metric = metric

    @property
    def metric(self) -> String:
        """
        A getter method that returns the metric attribute.

        :param: instance of the class.
        :return: metric.
        """
        return self._metric

    @metric.setter
    def metric(self, value: String) -> Void:
        """
        A setter method that changes the value of the metric attribute.

        :param value: a string that specifies the name of the distance method.
        :return: Void.
        """
        self._metric = value

    @classmethod
    def compute(self, P: NumpyArray) -> NumpyArray:
        """Compute the Correlation distance."""
        m, n = P.shape[: 2]
        mOnes = np.ones((m, n), dtype=np.float64)
        Pmean = np.mean(P, axis=1).reshape(-1, 1)
        P = P - Pmean * mOnes
        Pnorm = np.sqrt(np.sum(P ** 2, axis=1)).reshape(-1, 1) * mOnes
        P = P / Pnorm
        D = np.zeros((1, int(m * (m - 1) / 2)), dtype=np.float64)
        k = 0
        for i in range(m):
            d = np.zeros((m - i - 1, 1), dtype=np.float64)
            for q in range(n):
                d = d + (P[i, q] * P[i + 1: m, q]).reshape(-1, 1)
            d = np.where(d > 1, 1, d)
            D[0, k: k + m - i - 1] = (1 - d).reshape(-1)
            k += m - i - 1
        return D[0]

    def __repr__(self) -> String:
        """Custom repr function."""
        return f"{self.__dict__}"


class SpearmanCorrelation(PairwiseDistanceDescriptor):
    """Spearman rank correlation Distance."""

    _metric = String("metric")

    def __init__(self, metric: String = "Spearman Distance") -> Void:
        """Return an instance of SpearmanCorrelation class."""
        self._metric = metric

    @property
    def metric(self) -> String:
        """
        A getter method that returns the metric attribute.

        :param: instance of the class.
        :return: metric.
        """
        return self._metric

    @metric.setter
    def metric(self, value: String) -> Void:
        """
        A setter method that changes the value of the metric attribute.

        :param value: a string that specifies the name of the distance method.
        :return: Void.
        """
        self._metric = value

    @staticmethod
    def tiedrank(P, dim=None):
        """A helper method that computes the tiedrank for a given matrix."""
        if dim is None:
            dim = np.min(np.nonzero(P.shape)[0])
        Q = np.argsort(P, axis=dim)
        r1 = np.argsort(Q, axis=dim)
        Q = np.argsort(-P, axis=dim)
        r2 = np.argsort(Q, axis=dim)
        r2 = P.shape[dim] - r2
        return ((r1 + r2) / 2 + 1).astype(np.uint32)

    @classmethod
    def compute(self, P: NumpyArray) -> NumpyArray:
        """Compute the Spearman Correlation distance."""
        m, n = P.shape[: 2]
        mOnes = np.ones((m, n), dtype=np.float64)
        P = SpearmanCorrelation.tiedrank(P.T).T
        P = P - (n + 1) / 2
        Pnorm = np.sqrt(np.sum(P ** 2, axis=1)).reshape(-1, 1) * mOnes
        P = P / Pnorm
        D = np.zeros((1, int(m * (m - 1) / 2)), dtype=np.float64)
        k = 0
        for i in range(m):
            d = np.zeros((m - i - 1, 1), dtype=np.float64)
            for q in range(n):
                d = d + (P[i, q] * P[i + 1: m, q]).reshape(-1, 1)
            d = np.where(d > 1, 1, d)
            D[0, k: k + m - i - 1] = (1 - d).reshape(-1)
            k += m - i - 1
        return D[0]

    def __repr__(self) -> String:
        """Custom repr function."""
        return f"{self.__dict__}"


class Hamming(PairwiseDistanceDescriptor):
    """Pairwise Hamming Distance."""

    _metric = String("metric")

    def __init__(self, metric: String = "Hamming Distance") -> Void:
        """Return an instance of a Hamming class."""
        self._metric = metric

    @property
    def metric(self) -> String:
        """
        A getter method that returns the metric attribute.

        :param: instance of the class.
        :return: metric.
        """
        return self._metric

    @metric.setter
    def metric(self, value: String) -> Void:
        """
        A setter method that changes the value of the metric attribute.

        :param value: a string that specifies the name of the distance method.
        :return: Void.
        """
        self._metric = value

    @classmethod
    def compute(self, P: NumpyArray) -> NumpyArray:
        """Compute the Hamming distance."""
        m, n = P.shape[: 2]
        D = np.zeros((1, int(m * (m - 1) / 2)), dtype=np.float64)
        k = 0
        for i in range(m):
            nesum = np.zeros((m - i - 1, 1), dtype=np.float64)
            for q in range(n):
                nesum = nesum + np.not_equal(P[i, q], P[i + 1: m, q]).reshape(-1, 1)
            D[0, k: k + m - i - 1] = (nesum / n).reshape(-1)
            k += m - i - 1
        return D[0]

    def __repr__(self) -> String:
        """Custom repr function."""
        return f"{self.__dict__}"


class Jaccard(PairwiseDistanceDescriptor):
    """Pairwise Jaccard Distance."""

    _metric = String("metric")

    def __init__(self, metric: String = "Jaccard Distance") -> Void:
        """Return an instance of a Jaccard class."""
        self._metric = metric

    @property
    def metric(self) -> String:
        """
        A getter method that returns the metric attribute.

        :param: instance of the class.
        :return: metric.
        """
        return self._metric

    @metric.setter
    def metric(self, value: String) -> Void:
        """
        A setter method that changes the value of the metric attribute.

        :param value: a string that specifies the name of the distance method.
        :return: Void.
        """
        self._metric = value

    @classmethod
    def compute(self, P: NumpyArray) -> NumpyArray:
        """Compute the Jaccard distance."""
        m, n = P.shape[: 2]
        D = np.zeros((1, int(m * (m - 1) / 2)), dtype=np.float64)
        k = 0
        for i in range(m):
            nesum = np.zeros((m - i - 1, 1), dtype=np.float64)
            nzsum = np.zeros((m - i - 1, 1), dtype=np.float64)
            for q in range(n):
                nz = np.logical_or(np.not_equal(P[i, q], 0), np.not_equal(P[i + 1: m, q], 0)).reshape(-1, 1)
                ne = np.not_equal(P[i, q], P[i + 1: m, q]).reshape(-1, 1)
                nzsum = nzsum + nz
                nesum = nesum + np.logical_and(nz, ne)
            D[0, k: k + m - i - 1] = (nesum / nzsum).reshape(-1)
            k += m - i - 1
        return D[0]

    def __repr__(self) -> String:
        """Custom repr function."""
        return f"{self.__dict__}"


class pdist1(object):
    """
    An interface for distance calculation between each pair of points.

    The method of computation will be chosen based on the metric.
    """

    _metric = String("metric")
    _metric_distance = {'euclidean': Euclidean,
                        'default': Euclidean,
                        'seuclidean': StandardizedEuclidean,
                        'cityblock': CityBlock,
                        'mahalanobis': Mahalanobis,
                        'minkowski': Minkowski,
                        'chebyshev': Chebychev,
                        'cosine': Cosine,
                        'correlation': Correlation,
                        'spearman': SpearmanCorrelation,
                        'hamming': Hamming,
                        'jaccard': Jaccard,
                        }

    @property
    def metric(self) -> String:
        """
        A getter method that returns the metric attribute.

        :param: instance of the class.
        :return: metric.
        """
        return self._metric

    @metric.setter
    def metric(self, value: String) -> Void:
        """
        A setter method that changes the value of the metric attribute.

        :param value: a string that specifies the name of the distance method.
        :return: Void.
        """
        self._metric = value

    def __new__(self, P: NumpyArray, metric: String = "default", exp: PositiveInteger = 2) -> NumpyArray:
        """Compute the distance between P and Q based on metric."""
        if metric == 'minkowski':
            dist_object = Minkowski()
            return dist_object.compute(P, exp)
        else:
            dist_object = self._metric_distance.get(metric, Euclidean)
            return dist_object().compute(P)

    def __repr__(self) -> String:
        """Custom repr function."""
        return f"{self.__dict__}"


class pdist2(object):
    """
    An interface for distance calculation between each pair of two vectors.

    The method of computation will be chosen based on the metric.
    """

    _metric = String("metric")
    _metric_distance = {'manhattan': L1Distance,
                        'sqeuclidean': SquaredEuclideanDistance,
                        'euclidean': EuclideanDistance,
                        'default': EuclideanDistance,
                        'chi-squared': ChiSquaredDistance,
                        'cosine': CosineDistance,
                        'earthmover': EarthMoversDistance,
                        }

    @property
    def metric(self) -> String:
        """
        A getter method that returns the metric attribute.

        :param: instance of the class.
        :return: metric.
        """
        return self._metric

    @metric.setter
    def metric(self, value: String) -> Void:
        """
        A setter method that changes the value of the metric attribute.

        :param value: a string that specifies the name of the distance method.
        :return: Void.
        """
        self._metric = value

    def __new__(self, P: NumpyArray, Q: NumpyArray,
                metric: String = "default") -> NumpyArray:
        """Compute the distance between P and Q based on metric."""
        dist_object = self._metric_distance.get(metric, EuclideanDistance)
        return dist_object().compute(P, Q)

    def __repr__(self) -> String:
        """Custom repr function."""
        return f"{self.__dict__}"
