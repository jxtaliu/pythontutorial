#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

a = np.array([[1,  2],  [3,  4]])
print(a)

b = np.array([[3,  0],  [0,  6]])
print(b)

c = a * b
# d = np.matmul(a, b)
d = np.dot(a, b)
print(c)
print(d)
