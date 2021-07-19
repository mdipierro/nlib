"""
Code from book "Annotated Algorithms in Python3"
Written by Massimo Di Pierro - BSD License
"""
from random import choice

import mincemeat

strings = ["".join(choice("ATGC") for i in range(10)) for j in range(100)]


def mapfn(k1, string):
    yield ("ACCA" in string, 1)


def reducefn(k2, values):
    return len(values)


s = mincemeat.Server()
s.mapfn = mapfn
s.reducefn = reducefn
s.datasource = dict(enumerate(strings))
results = s.run_server(password="changeme")
print(results)
