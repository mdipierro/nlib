
"""
Code from book "Annotated Algorithms in Python3"
Written by Massimo Di Pierro - BSD License
"""
import random

from psim import PSim


def scalar_product_test1(n, p):
    comm = PSim(p)
    h = n/p
    if comm.rank == 0:
        a = [random.random() for i in range(n)]
        b = [random.random() for i in range(n)]
        for k in range(1, p):
            comm.send(k, a[k * h:k * h+h])
            comm.send(k, b[k * h:k * h+h])
    else:
        a = comm.recv(0)
        b = comm.recv(0)
    scalar = sum(a[i] * b[i] for i in range(h))
    if comm.rank == 0:
        for k in range(1, p):
            scalar += comm.recv(k)
        print(scalar)
    else:
        comm.send(0, scalar)

scalar_product_test(10, 2)

