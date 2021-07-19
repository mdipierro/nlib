
"""
Code from book "Annotated Algorithms in Python3"
Written by Massimo Di Pierro - BSD License
"""
import functools
import math
import os
import pickle
import time


def BUS(i, j):
    return True

def SWITCH(i, j):
    return True

def MESH1(p):
    return lambda i, j, p=p: (i - j) ** 2 == 1

def TORUS1(p):
    return lambda i, j, p=p: (i - j + p) % p == 1 or (j - i + p) % p == 1

def MESH2(p):
    q = int(math.sqrt(p) + 0.1)
    return lambda i, j, q=q: (
        (i % q - j % q) ** 2, (i / q - j / q) ** 2) in [(1, 0), (0, 1)]

def TORUS2(p):
    q = int(math.sqrt(p) + 0.1)
    return lambda i, j, q=q: (
        ((i % q - j % q + q) % q, (i / q - j / q + q) % q)
        in [(0, 1), (1, 0)]
        or
        ((j % q - i % q + q) % q, (j / q - i / q + q) % q)
        in [(0, 1), (1, 0)]
    )

def TREE(i, j):
    return i == int((j - 1) / 2) or j == int((i - 1) / 2)

class PSim(object):
    def __init__(self, p, topology=SWITCH, logfilename=None):
        """
        forks p-1 processes and creates p * p
        """
        self.logfile = logfilename and open(logfilename, "w")
        self.topology = topology
        self.log("START: creating %i parallel processes\n" % p)
        self.nprocs = p
        self.pipes = {}
        for i in range(p):
            for j in range(p):
                self.pipes[i, j] = os.pipe()
        self.rank = 0
        for i in range(1, p):
            if not os.fork():
                self.rank = i
                break
        self.log("START: done.\n")

    def log(self, message):
        """
        logs the message into self._logfile
        """
        if self.logfile != None:
            self.logfile.write(message)

    def send(self, j, data):
        """
        sends data to process #j
        """
        if not self.topology(self.rank, j):
            raise RuntimeError("topology violation")
        self._send(j, data)

    def _send(self, j, data):
        """
        sends data to process #j ignoring topology
        """
        if j < 0 or j >= self.nprocs:
            self.log("process %i: send(%i, ...) failed!\n" %
                     (self.rank, j))
            raise Exception
        self.log("process %i: send(%i, %s) starting...\n" %
                 (self.rank, j, repr(data)))
        s = pickle.dumps(data)
        os.write(self.pipes[self.rank, j][1], str.zfill(str(len(s)), 10))
        os.write(self.pipes[self.rank, j][1], s)
        self.log("process %i: send(%i, %s) success.\n" %
                 (self.rank, j, repr(data)))

    def recv(self, j):
        """
        returns the data received from process #j
        """
        if not self.topology(self.rank, j):
            raise RuntimeError("topology violation")
        return self._recv(j)

    def _recv(self, j):
        """
        returns the data received from process #j ignoring topology
        """
        if j < 0 or j >= self.nprocs:
            self.log("process %i: recv(%i) failed!\n" % (self.rank, j))
            raise RuntimeError
        self.log("process %i: recv(%i) starting...\n" % (self.rank, j))
        try:
            size = int(os.read(self.pipes[j, self.rank][0], 10))
            s = os.read(self.pipes[j, self.rank][0], size)
        except Exception as e:
            self.log("process %i: COMMUNICATION ERROR!!!\n" % (self.rank))
            raise e
        data = pickle.loads(s)
        self.log("process %i: recv(%i) done.\n" % (self.rank, j))
        return data


    def one2all_broadcast(self, source, value):
        self.log(
            "process %i: BEGIN one2all_broadcast(%i, %s)\n"
            % (self.rank, source, repr(value))
        )
        if self.rank == source:
            for i in range(0, self.nprocs):
                if i != source:
                    self._send(i, value)
        else:
            value = self._recv(source)
        self.log(
            "process %i: END one2all_broadcast(%i, %s)\n"
            % (self.rank, source, repr(value))
        )
        return value

    def all2all_broadcast(self, value):
        self.log("process %i: BEGIN all2all_broadcast(%s)\n" %
                 (self.rank, repr(value)))
        vector = self.all2one_collect(0, value)
        vector = self.one2all_broadcast(0, vector)
        self.log("process %i: END all2all_broadcast(%s)\n" %
                 (self.rank, repr(value)))
        return vector


    def one2all_scatter(self, source, data):
        self.log(
            "process %i: BEGIN all2one_scatter(%i, %s)\n"
            % (self.rank, source, repr(data))
        )
        if self.rank == source:
            h, remainder = divmod(len(data), self.nprocs)
            if remainder:
                h += 1
            for i in range(self.nprocs):
                self._send(i, data[i * h : i * h + h])
        vector = self._recv(source)
        self.log(
            "process %i: END all2one_scatter(%i, %s)\n"
            % (self.rank, source, repr(data))
        )
        return vector

    def all2one_collect(self, destination, data):
        self.log(
            "process %i: BEGIN all2one_collect(%i, %s)\n"
            % (self.rank, destination, repr(data))
        )
        self._send(destination, data)
        if self.rank == destination:
            vector = [self._recv(i) for i in range(self.nprocs)]
        else:
            vector = []
        self.log(
            "process %i: END all2one_collect(%i, %s)\n"
            % (self.rank, destination, repr(data))
        )
        return vector


    def all2one_reduce(self, destination, value, op=lambda a, b: a + b):
        self.log("process %i: BEGIN all2one_reduce(%s)\n" %
                 (self.rank, repr(value)))
        self._send(destination, value)
        if self.rank == destination:
            result = functools.reduce(
                op, [self._recv(i) for i in range(self.nprocs)])
        else:
            result = None
        self.log("process %i: END all2one_reduce(%s)\n" %
                 (self.rank, repr(value)))
        return result

    def all2all_reduce(self, value, op=lambda a, b: a + b):
        self.log("process %i: BEGIN all2all_reduce(%s)\n" %
                 (self.rank, repr(value)))
        result = self.all2one_reduce(0, value, op)
        result = self.one2all_broadcast(0, result)
        self.log("process %i: END all2all_reduce(%s)\n" %
                 (self.rank, repr(value)))
        return result

    @staticmethod
    def sum(x, y):
        return x + y

    @staticmethod
    def mul(x, y):
        return x * y

    @staticmethod
    def max(x, y):
        return max(x, y)

    @staticmethod
    def min(x, y):
        return min(x, y)

    def barrier(self):
        self.log("process %i: BEGIN barrier()\n" % (self.rank))
        self.all2all_broadcast(0)
        self.log("process %i: END barrier()\n" % (self.rank))
        return

