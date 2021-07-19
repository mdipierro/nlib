"""
Code from book "Annotated Algorithms in Python3"
Written by Massimo Di Pierro - BSD License
"""
import random

from nlib import *


class NetworkReliability(MCEngine):
    def __init__(self, n_nodes, start, stop):
        self.links = []
        self.n_nodes = n_nodes
        self.start = start
        self.stop = stop

    def add_link(self, i, j, failure_probability):
        self.links.append((i, j, failure_probability))

    def simulate_once(self):
        nodes = DisjointSets(self.n_nodes)
        for i, j, pf in self.links:
            if random.random() > pf:
                nodes.join(i, j)
        return nodes.joined(i, j)


def main():
    s = NetworkReliability(100, start=0, stop=1)
    for k in range(300):
        s.add_link(random.randint(0, 99), random.randint(0, 99), random.random())
    print(s.simulate_many())


main()
