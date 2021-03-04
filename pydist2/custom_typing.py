#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
| This is a customized script that defines a customized type hinting.

| This program and the accompanying materials are made available under the terms of the `MIT License`_.
| SPDX short identifier: MIT
| Contributors:
    Mahmoud Harmouch, mail_.

.. _MIT License: https://opensource.org/licenses/MIT
.. _mail: mahmoudddharmouchhh@gmail.com
"""
import numpy as np


class TypeDescriptor(object):
    """
    A basic Type Descriptor class that allows customize handling for different attributes.

    It intercepts get, set and repr methods.
    """

    def __init__(self, name=None):
        """
        An init function that initialises and stores the name of the data.

        :param name: object type which is a key stored in the instance dictionnary.
        """
        self.name = name

    def __set__(self, instance, value):
        """A setter method that store the value of the name attribute in the instance dictionnary."""
        instance.__dict__[self.name] = value

    def __get__(self, instance, cls):
        """A getter method that fetch the value of the name attribute from the instance dictionnary."""
        return instance.__dict__[self.name]

    def __repr__(self):
        """A delete method that removes the value of the name attribute from the instance dictionnary."""
        return str(self.__dict__)


class CustomType(TypeDescriptor):
    """A custom tyoe class that implements Descriptor."""

    _type = object

    def __set__(self, instance, value):
        """A setter method that defines the type of the _type attribute."""
        if not isinstance(value, self._type):
            raise TypeError('Expected {0}, given {1}!'
                            .format(self._type, type(value)))
        super().__set__(instance, value)


class Integer(CustomType):
    """A customized Integer data type."""

    _type = int


class Float(CustomType):
    """A customized Float data type."""

    _type = float


class String(CustomType):
    """A customized String data type."""

    _type = str


class NumpyArray(CustomType):
    """A customized NumpyArray data type."""

    _type = np.ndarray


class Void(CustomType):
    """A customized Void data type."""

    _type = None


class Positive(TypeDescriptor):
    """A customized Positive data type."""

    def __set__(self, instance, value):
        """A customized Void data type."""
        if value < 0:
            raise ValueError('Expected value to be >= 0')
        super().__set__(instance, value)


class PositiveInteger(Integer, Positive):
    """A customized Positive Integer data type."""

    pass
