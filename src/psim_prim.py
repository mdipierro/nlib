
"""
Code from book "Annotated Algorithms in Python3"
Written by Massimo Di Pierro - BSD License
"""
import random

from psim import PSim


def random_adjacency_matrix(n):
    A = []
    for r in range(n):
        A.append([0] * n)
    for r in range(n):
        for c in range(0, r):
            A[r][c] = A[c][r] = random.randint(1, 100)
    return A

class Vertex(object):
    def __init__(self, path=[0, 1, 2]):
        self.path = path

def weight(path=[0, 1, 2], adjacency=None):
    return sum(adjacency[path[i-1]][path[i]] for i in range(1, len(path)))

def bb(adjacency, p=1):
    n = len(adjacency)
    comm = PSim(p)
    Q = []
    path = [0]
    Q.append(Vertex(path))
    bound = float("inf')
    optimal = None
    local_vertices = comm.one2all_scatter(0, range(n))
    while True:
        if comm.rank == 0:
            vertex = Q.pop() if Q else None
        else:
            vertex = None
        vertex = comm.one2all_broadcast(0, vertex)
        if vertex is None:
            break
        P = []
        for k in local_vertices:
            if not k in vertex.path:
                new_path = vertex.path+[k]
                new_path_length = weight(new_path, adjacency)
                if new_path_length<bound:
                    if len(new_path) == n:
                        new_path.append(new_path[0])
                        new_path_length = weight(new_path, adjacency)
                        if new_path_length<bound:
                            bound = new_path_length  # bcast
                            optimal = new_path       # bcast
                    else:
                        new_vertex = Vertex(new_path)
                        P.append(new_vertex)  # fix this
                print(new_path, new_path_length)
        x = (bound, optimal)
        x = comm.all2all_reduce(x, lambda a, b: min(a, b))
        (bound, optimal) = x

        P = comm.all2one_collect(0, P)
        if comm.rank == 0:
            for item in P:
                Q+=item
    return optimal, bound


m = random_adjacency_matrix(5)
print(bb(m, p=2))

