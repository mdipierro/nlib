
"""
Code from book "Annotated Algorithms in Python3"
Written by Massimo Di Pierro - BSD License
"""
import random

from psim import PSim


def mergesort(A, x=0, z=None):
    if z is None: z = len(A)
    if x<z-1:
        y = int((x+z)/2)
        mergesort(A, x, y)
        mergesort(A, y, z)
        merge(A, x, y, z)

def merge(A, x, y, z):
    B, i, j = [], x, y
    while True:
        if A[i]<=A[j]:
            B.append(A[i])
            i=i+1
        else:
            B.append(A[j])
            j=j+1
        if i == y:
            while j<z:
                B.append(A[j])
                j=j+1
            break
        if j == z:
            while i<y:
                B.append(A[i])
                i=i+1
            break
    A[x:z]=B

def mergesort_test(n, p):
    comm = PSim(p)
    if comm.rank == 0:
        data = [random.random() for i in range(n)]
        comm.send(1, data[n/2:])
        mergesort(data, 0, n/2)
        data[n/2:] = comm.recv(1)
        merge(data, 0, n/2, n)
        print(data)
    else:
        data = comm.recv(0)
        mergesort(data)
        comm.send(0, data)

mergesort_test(20, 2)

