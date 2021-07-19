
"""
Code from book "Annotated Algorithms in Python3"
Written by Massimo Di Pierro - BSD License
"""
import random

from psim import PSim


def scalar_product_test2(n, p):
    comm = PSim(p)
    a = b = None
    if comm.rank == 0:
        a = [random.random() for i in range(n)]
        b = [random.random() for i in range(n)]
    a = comm.one2all_scatter(0, a)
    b = comm.one2all_scatter(0, b)

    scalar = sum(a[i] * b[i] for i in range(len(a)))

    scalar = comm.all2one_reduce(0, scalar)
    if comm.rank == 0:
        print(scalar)

scalar_product_test2(10, 2)

