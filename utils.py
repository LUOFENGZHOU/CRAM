#!/usr/bin/env python
# -*- coding: utf8 -*-
"""Helper methods that are used in the project.
"""
from __future__ import division
from math import log


def lg(x):
    res = 0
    try:
        res = log(x, 2)
    except ValueError:
        pass
    return res


def exp(x):
    return 2 ** x


def reverse_argsort(X):
    indices = range(len(X))
    indices.sort(key=X.__getitem__, reverse=True)
    return indices
