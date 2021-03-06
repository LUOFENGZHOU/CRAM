#!/usr/bin/env python
# -*- coding: utf8 -*-
"""Computes the SNML code for a sequence. We use log-sum-exp trick  to fix the math overflow/underflow error.
https://stats.stackexchange.com/questions/105602/example-of-how-the-log-sum-exp-trick-works-in-naive-bayes
"""
from __future__ import division
from collections import Counter
from functools import partial
import sys

import numpy as np

from utils import lg, exp


"""
code provided by "Causal Inference on Event Sequences"
"""


def regret(t1, t, x):
    # define a convex compact subset to get a finite regret
    # epsilon = 0.0001, 1-epsilon = 0.9999
    # if not t1:
    #     t1 = 1
    #     t = 10000    # epsilon = 0.0001
    # elif t1 == t - 1:
    #     t1 = 9999
    #     t = 10000    # 1-epsilon = 0.9999
    t0 = t - 1 - t1
    ll1 = (t1 + 1) * lg(t1 + 1) + t0 * lg(t0)
    ll0 = t1 * lg(t1) + (t0 + 1) * lg(t0 + 1)
    max_ll = max(ll1, ll0)
    lg_numer = ll1 if x == 1 else ll0
    lg_denom = max_ll + lg(exp(ll1 - max_ll) + exp(ll0 - max_ll))
    return lg_denom - lg_numer


def bernoulli(X):
    # todo(kailash): fix for extrema
    # for t=0,fix it
    t1 = 0
    res = 0.0
    for t, x in enumerate(X, 1):
        if t == 1:
            t1 += int(x == 1)
            continue
        res += regret(t1, t, x)
        t1 += int(x == 1)
    return res


def cbernoulli(X, Y):
    res = 0.0
    t_x, t_y, t_ones = 0, 0, 0
    for t, x in enumerate(X, 1):
        y = Y[t - 1]
        if t == 1:
            t_x += int(x == 1)
            t_y += int(y == 1)
            t_max = t_x or t_y
            res += regret(0, t, x)
            continue

        t1 = min(t_x, t_y) if x == 0 else t_max
        res += regret(t1, t, x)

        t_x += int(x == 1)
        t_y += int(y == 1)
        t_max += int(x == 1) or int(y == 1)
    # assert res <= bernoulli(X)
    return res


def multinomial(X):
    regret = 0.0
    freqs = Counter()
    size = len(X)
    supp = set(X)

    def lg_reduction(x, to_predict, n):
        freq = freqs[x] + 1 if to_predict == x else freqs[x]
        return freq * (lg(freq) - lg(n))

    for n, x in enumerate(X):
        if not n:
            freqs.update([x])
            continue

        lls = []
        lg_numer = None
        for k in supp:
            ll_k = sum(map(partial(lg_reduction, to_predict=k, n=n), supp))
            if k == x:
                lg_numer = ll_k
            lls.append(ll_k)

        max_ll = max(lls)
        lg_denom = max_ll + lg(sum(exp(ll_k - max_ll) for ll_k in lls))
        regret += - lg_numer + lg_denom
        freqs.update([x])

    return regret


def cmultinomial(X, Y):
    regret = 0.0
    freqs = Counter()
    size = len(X)
    supp = set(X)

    def lg_reduction(x, to_predict, n):
        freq = freqs[x] + 1 if to_predict == x else freqs[x]
        return freq * (lg(freq) - lg(n))

    for n, x in enumerate(X):
        if not n:
            freqs.update([x])
            continue

        lls = []
        lg_numer = None
        for k in supp:
            ll_k = sum(map(partial(lg_reduction, to_predict=k, n=n), supp))
            if k == x:
                lg_numer = ll_k
            lls.append(ll_k)

        max_ll = max(lls)
        lg_denom = max_ll + lg(sum(exp(ll_k - max_ll) for ll_k in lls))
        regret += - lg_numer + lg_denom
        freqs.update([x])

    return regret


if __name__ == "__main__":
    print bernoulli([0] * 20)
    print bernoulli([1] * 20)
    print cbernoulli([0] * 20, [1] * 20)
    print cbernoulli([1] * 20, [0] * 20)
    print bernoulli([0,0,0,0,0,0,1])
    X = np.random.choice([1, 1, 1, 1, 1, 1, 1, 0, 0], 5000)
    Y = np.roll(X, 3)
    print cbernoulli(X, Y)
