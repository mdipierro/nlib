import cmath
import copy
import io
import math
import os
import pickle
import random
import sqlite3
import tempfile
import time
import unittest
import uuid
from heapq import heapify, heappop, heappush

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse

__all__ = [
    "CONSTANT",
    "CUBIC",
    "Canvas",
    "Cholesky",
    "Cluster",
    "D",
    "DD",
    "Dijkstra",
    "DisjointSets",
    "E",
    "Ellipse",
    "Figure",
    "FigureCanvasAgg",
    "Jacobi_eigenvalues",
    "Kruskal",
    "LINEAR",
    "MCEngine",
    "MCG",
    "Markowitz",
    "MarsenneTwister",
    "Matrix",
    "NeuralNetwork",
    "POLYNOMIAL",
    "PersistentDictionary",
    "Prim",
    "PrimVertex",
    "QUADRATIC",
    "QUARTIC",
    "QuadratureIntegrator",
    "RandomSource",
    "Trader",
    "fake_stock_prices",
    "bootstrap",
    "breadth_first_search",
    "compute_correlation_matrix",
    "condition_number",
    "confidence_intervals",
    "continuum_knapsack",
    "correlation",
    "covariance",
    "decode_huffman",
    "depth_first_search",
    "encode_huffman",
    "fib",
    "fit",
    "fit_least_squares",
    "gradient",
    "hessian",
    "integrate",
    "integrate_naive",
    "integrate_quadrature_naive",
    "invert_bicgstab",
    "invert_minimum_residual",
    "is_almost_symmetric",
    "is_almost_zero",
    "is_positive_definite",
    "jacobian",
    "lcs",
    "leapfrog",
    "make_maze",
    "mean",
    "memoize",
    "memoize_persistent",
    "needleman_wunsch",
    "norm",
    "optimize_bisection",
    "optimize_golden_search",
    "optimize_newton",
    "optimize_newton_multi",
    "optimize_newton_multi_imporved",
    "optimize_secant",
    "partial",
    "resample",
    "sd",
    "solve_bisection",
    "solve_fixed_point",
    "solve_newton",
    "solve_newton_multi",
    "solve_secant",
    "variance",
    "mycos",
    "mysin",
    "myexp",
    "exp",
]

"""
Code from book "Annotated Algorithms in Python3"
Written by Massimo Di Pierro - BSD License
"""


class PersistentDictionary(object):
    """
    A sqlite based key, value storage.
    The value can be any pickleable object.
    Similar interface to Python dict
    Supports the GLOB syntax in methods keys(), items(), __delitem__()

    Usage Example:
    >>> p = PersistentDictionary(path="test.sqlite")
    >>> key = "test/" + p.uuid()
    >>> p[key] = {"a": 1, "b": 2}
    >>> p[key]
    {'a': 1, 'b': 2}
    >>> len(p.keys("test/*"))
    1
    >>> del p[key]
    """

    CREATE_TABLE = "CREATE TABLE persistence (pkey, pvalue)"
    SELECT_KEYS = "SELECT pkey FROM persistence WHERE pkey GLOB ?"
    SELECT_VALUE = "SELECT pvalue FROM persistence WHERE pkey GLOB ?"
    INSERT_KEY_VALUE = "INSERT INTO persistence(pkey, pvalue) VALUES (?, ?)"
    UPDATE_KEY_VALUE = "UPDATE persistence SET pvalue = ? WHERE pkey = ?"
    DELETE_KEY_VALUE = "DELETE FROM persistence WHERE pkey LIKE ?"
    SELECT_KEY_VALUE = "SELECT pkey, pvalue FROM persistence WHERE pkey GLOB ?"

    def __init__(self, path="persistence.sqlite", autocommit=True, serializer=pickle):
        self.path = path
        self.autocommit = autocommit
        self.serializer = serializer
        create_table = not os.path.exists(path)
        self.connection = sqlite3.connect(path)
        self.connection.text_factory = str  # do not use unicode
        self.cursor = self.connection.cursor()
        if create_table:
            self.cursor.execute(self.CREATE_TABLE)
            self.connection.commit()

    def uuid(self):
        return str(uuid.uuid4())

    def keys(self, pattern=" * "):
        "returns a list of keys filtered by a pattern, * is the wildcard"
        self.cursor.execute(self.SELECT_KEYS, (pattern,))
        return [row[0] for row in self.cursor.fetchall()]

    def __contains__(self, key):
        return True if self.get(key) != None else False

    def __iter__(self):
        for key in self:
            yield key

    def __setitem__(self, key, value):
        if key in self:
            if value is None:
                del self[key]
            else:
                svalue = self.serializer.dumps(value)
                self.cursor.execute(self.UPDATE_KEY_VALUE, (svalue, key))
        else:
            svalue = self.serializer.dumps(value)
            self.cursor.execute(self.INSERT_KEY_VALUE, (key, svalue))
        if self.autocommit:
            self.connection.commit()

    def get(self, key):
        self.cursor.execute(self.SELECT_VALUE, (key,))
        row = self.cursor.fetchone()
        return self.serializer.loads(row[0]) if row else None

    def __getitem__(self, key):
        self.cursor.execute(self.SELECT_VALUE, (key,))
        row = self.cursor.fetchone()
        if not row:
            raise KeyError
        return self.serializer.loads(row[0])

    def __delitem__(self, pattern):
        self.cursor.execute(self.DELETE_KEY_VALUE, (pattern,))
        if self.autocommit:
            self.connection.commit()

    def items(self, pattern=" * "):
        self.cursor.execute(self.SELECT_KEY_VALUE, (pattern,))
        return [
            (row[0], self.serializer.loads(row[1])) for row in self.cursor.fetchall()
        ]

    def dumps(self, pattern=" * "):
        self.cursor.execute(self.SELECT_KEY_VALUE, (pattern,))
        rows = self.cursor.fetchall()
        return self.serializer.dumps(
            dict((row[0], self.serializer.loads(row[1])) for row in rows)
        )

    def loads(self, raw):
        data = self.serializer.loads(raw)
        for key, value in data.iteritems():
            self[key] = value


os.environ["MPLCONfigureDIR"] = tempfile.mkdtemp()


class Canvas(object):
    def __init__(self, title="", xlab="x", ylab="y", xrange=None, yrange=None):
        self.fig = Figure()
        self.fig.set_facecolor("white")
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title(title)
        self.ax.set_xlabel(xlab)
        self.ax.set_ylabel(ylab)
        if xrange:
            self.ax.set_xlim(xrange)
        if yrange:
            self.ax.set_ylim(yrange)
        self.legend = []

    def save(self, filename="plot.png"):
        if self.legend:
            legend = self.ax.legend(
                [e[0] for e in self.legend], [e[1] for e in self.legend]
            )
            legend.get_frame().set_alpha(0.7)
        if filename:
            FigureCanvasAgg(self.fig).print_png(open(filename, "wb"))
        else:
            s = io.StringIO()
            FigureCanvasAgg(self.fig).print_png(s)
            return s.getvalue()

    def binary(self):
        return self.save(None)

    def hist(self, data, bins=20, color="blue", legend=None):
        q = self.ax.hist(data, bins)
        # if legend:
        #    self.legend.append((q[0], legend))
        return self

    def plot(self, data, color="blue", style="-", width=2, legend=None, xrange=None):
        if callable(data) and xrange:
            x = [xrange[0] + 0.01 * i * (xrange[1] - xrange[0]) for i in range(0, 101)]
            y = [data(p) for p in x]
        elif data and isinstance(data[0], (int, float)):
            x, y = range(len(data)), data
        else:
            x, y = [p[0] for p in data], [p[1] for p in data]
        q = self.ax.plot(x, y, linestyle=style, linewidth=width, color=color)
        if legend:
            self.legend.append((q[0], legend))
        return self

    def errorbar(self, data, color="black", marker="o", width=2, legend=None):
        x, y, dy = [p[0] for p in data], [p[1] for p in data], [p[2] for p in data]
        q = self.ax.errorbar(x, y, yerr=dy, fmt=marker, linewidth=width, color=color)
        if legend:
            self.legend.append((q[0], legend))
        return self

    def ellipses(self, data, color="blue", width=0.01, height=0.01, legend=None):
        for point in data:
            x, y = point[:2]
            dx = point[2] if len(point) > 2 else width
            dy = point[3] if len(point) > 3 else height
            ellipse = Ellipse(xy=(x, y), width=dx, height=dy)
            self.ax.add_artist(ellipse)
            ellipse.set_clip_box(self.ax.bbox)
            ellipse.set_alpha(0.5)
            ellipse.set_facecolor(color)
        if legend:
            self.legend.append((q[0], legend))
        return self

    def imshow(self, data, interpolation="bilinear"):
        self.ax.imshow(data).set_interpolation(interpolation)
        return self


class memoize(object):
    def __init__(self, f):
        self.f = f
        self.storage = {}

    def __call__(self, *args, **kwargs):
        key = str((self.f.__name__, args, kwargs))
        try:
            value = self.storage[key]
        except KeyError:
            value = self.f(*args, **kwargs)
            self.storage[key] = value
        return value


@memoize
def fib(n):
    return n if n < 2 else fib(n - 1) + fib(n - 2)


class memoize_persistent(object):
    STORAGE = "memoize.sqlite"

    def __init__(self, f):
        self.f = f
        self.storage = PersistentDictionary(memoize_persistent.STORAGE)

    def __call__(self, *args, **kwargs):
        key = str((self.f.__name__, args, kwargs))
        if key in self.storage:
            value = self.storage[key]
        else:
            value = self.f(*args, **kwargs)
            self.storage[key] = value
        return value


def timef(f, ns=1000, dt=60):
    t = t0 = time.time()
    for k in range(1, ns):
        f()
        t = time.time()
        if t - t0 > dt:
            break
    return (t - t0) / k


def breadth_first_search(graph, start):
    vertices, links = graph
    blacknodes = []
    graynodes = [start]
    neighbors = [[] for vertex in vertices]
    for link in links:
        neighbors[link[0]].append(link[1])
    while graynodes:
        current = graynodes.pop()
        for neighbor in neighbors[current]:
            if not neighbor in blacknodes + graynodes:
                graynodes.insert(0, neighbor)
        blacknodes.append(current)
    return blacknodes


def depth_first_search(graph, start):
    vertices, links = graph
    blacknodes = []
    graynodes = [start]
    neighbors = [[] for vertex in vertices]
    for link in links:
        neighbors[link[0]].append(link[1])
    while graynodes:
        current = graynodes.pop()
        for neighbor in neighbors[current]:
            if not neighbor in blacknodes + graynodes:
                graynodes.append(neighbor)
        blacknodes.append(current)
    return blacknodes


class DisjointSets(object):
    def __init__(self, n):
        self.sets = [-1] * n
        self.counter = n

    def parent(self, i):
        while True:
            j = self.sets[i]
            if j < 0:
                return i
            i = j

    def join(self, i, j):
        i, j = self.parent(i), self.parent(j)
        if i != j:
            self.sets[i] += self.sets[j]
            self.sets[j] = i
            self.counter -= 1
            return True  # they have been joined
        return False  # they were already joined

    def joined(self, i, j):
        return self.parent(i) == self.parent(j)

    def __len__(self):
        return self.counter


def make_maze(n, d):
    walls = [
        (i, i + n ** j)
        for i in range(n ** 2)
        for j in range(d)
        if (i / n ** j) % n + 1 < n
    ]
    torn_down_walls = []
    ds = DisjointSets(n ** d)
    random.shuffle(walls)
    for i, wall in enumerate(walls):
        if ds.join(wall[0], wall[1]):
            torn_down_walls.append(wall)
        if len(ds) == 1:
            break
    walls = [wall for wall in walls if not wall in torn_down_walls]
    return walls, torn_down_walls


def Kruskal(graph):
    vertices, links = graph
    A = []
    S = DisjointSets(len(vertices))
    links.sort(cmp=lambda a, b: cmp(a[2], b[2]))
    for source, dest, length in links:
        if S.join(source, dest):
            A.append((source, dest, length))
    return A


class PrimVertex(object):
    INFINITY = 1e100

    def __init__(self, id, links):
        self.id = id
        self.closest = None
        self.closest_dist = PrimVertex.INFINITY
        self.neighbors = [link[1:] for link in links if link[0] == id]

    def __lt__(self, other):
        return self.closest_dist < other.closest_dist


def Prim(graph, start):
    vertices, links = graph
    P = [PrimVertex(i, links) for i in vertices]
    Q = [P[i] for i in vertices if not i == start]
    vertex = P[start]
    while Q:
        for neighbor_id, length in vertex.neighbors:
            neighbor = P[neighbor_id]
            if neighbor in Q and length < neighbor.closest_dist:
                neighbor.closest = vertex
                neighbor.closest_dist = length
        heapify(Q)
        vertex = heappop(Q)
    return [(v.id, v.closest.id, v.closest_dist) for v in P if not v.id == start]


def Dijkstra(graph, start):
    vertices, links = graph
    P = [PrimVertex(i, links) for i in vertices]
    Q = [P[i] for i in vertices if not i == start]
    vertex = P[start]
    vertex.closest_dist = 0
    while Q:
        for neighbor_id, length in vertex.neighbors:
            neighbor = P[neighbor_id]
            dist = length + vertex.closest_dist
            if neighbor in Q and dist < neighbor.closest_dist:
                neighbor.closest = vertex
                neighbor.closest_dist = dist
        heapify(Q)
        vertex = heappop(Q)
    return [(v.id, v.closest.id, v.closest_dist) for v in P if not v.id == start]


class HuffmanNode:
    def __init__(self, frequency, content):
        self.frequency = frequency
        self.content = content

    def __lt__(self, other):
        return self.frequency < other.frequency

    def unpack(self):
        return self.frequency, self.content


def encode_huffman(input):
    def inorder_tree_walk(t, key, keys):
        (f, ab) = t
        if isinstance(ab, tuple):
            inorder_tree_walk(ab[0], key + "0", keys)
            inorder_tree_walk(ab[1], key + "1", keys)
        else:
            keys[ab] = key

    symbols = {}
    for symbol in input:
        symbols[symbol] = symbols.get(symbol, 0) + 1
    heap = []
    for (k, f) in symbols.items():
        heappush(heap, HuffmanNode(f, k))
    while len(heap) > 1:
        (f1, k1) = heappop(heap).unpack()
        (f2, k2) = heappop(heap).unpack()
        heappush(heap, HuffmanNode(f1 + f2, ((f1, k1), (f2, k2))))
    symbol_map = {}
    root = heap[0].unpack()
    inorder_tree_walk(root, "", symbol_map)
    encoded = "".join(symbol_map[symbol] for symbol in input)
    return symbol_map, encoded


def decode_huffman(keys, encoded):
    reversed_map = dict((v, k) for (k, v) in keys.items())
    i, output = 0, []
    for j in range(1, len(encoded) + 1):
        if encoded[i:j] in reversed_map:
            output.append(reversed_map[encoded[i:j]])
            i = j
    return "".join(output)


def lcs(a, b):
    previous = [0] * len(a)
    for i, r in enumerate(a):
        current = []
        for j, c in enumerate(b):
            if r == c:
                e = previous[j - 1] + 1 if i * j > 0 else 1
            else:
                e = max(previous[j] if i > 0 else 0, current[-1] if j > 0 else 0)
            current.append(e)
        previous = current
    return current[-1]


def needleman_wunsch(a, b, p=0.97):
    z = []
    for i, r in enumerate(a):
        z.append([])
        for j, c in enumerate(b):
            if r == c:
                e = z[i - 1][j - 1] + 1 if i * j > 0 else 1
            else:
                e = p * max(z[i - 1][j] if i > 0 else 0, z[i][j - 1] if j > 0 else 0)
            z[-1].append(e)
    return z


def continuum_knapsack(a, b, c):
    table = [(a[i] / b[i], i) for i in range(len(a))]
    table.sort()
    table.reverse()
    f = 0.0
    for (y, i) in table:
        quantity = min(c / b[i], 1)
        x.append((i, quantity))
        c = c - b[i] * quantity
        f = f + a[i] * quantity
    return (f, x)


class Cluster(object):
    def __init__(self, points, metric, weights=None):
        self.points, self.metric = points, metric
        self.k = len(points)
        self.w = weights or [1.0] * self.k
        self.q = dict((i, [i]) for i, e in enumerate(points))
        self.d = []
        for i in range(self.k):
            for j in range(i + 1, self.k):
                m = metric(points[i], points[j])
                if not m is None:
                    self.d.append((m, i, j))
        self.d.sort()
        self.dd = []

    def parent(self, i):
        while isinstance(i, int):
            (parent, i) = (i, self.q[i])
        return parent, i

    def step(self):
        if self.k > 1:
            # find new clusters to join
            (self.r, i, j), self.d = self.d[0], self.d[1:]
            # join them
            i, x = self.parent(i)  # find members of cluster i
            j, y = self.parent(j)  # find members if cluster j
            x += y  # join members
            self.q[j] = i  # make j cluster point to i
            self.k -= 1  # decrease cluster count
            # update all distances to new joined cluster
            new_d = []  # links not related to joined clusters
            old_d = {}  # old links related to joined clusters
            for (r, h, k) in self.d:
                if h in (i, j):
                    a, b = old_d.get(k, (0.0, 0.0))
                    old_d[k] = a + self.w[k] * r, b + self.w[k]
                elif k in (i, j):
                    a, b = old_d.get(h, (0.0, 0.0))
                    old_d[h] = a + self.w[h] * r, b + self.w[h]
                else:
                    new_d.append((r, h, k))
            new_d += [(a / b, i, k) for k, (a, b) in old_d.items()]
            new_d.sort()
            self.d = new_d
            # update weight of new cluster
            self.w[i] = self.w[i] + self.w[j]
            # get new list of cluster members
            self.v = [s for s in self.q.values() if isinstance(s, list)]
            self.dd.append((self.r, len(self.v)))
        return self.r, self.v

    def find(self, k):
        # if necessary start again
        if self.k < k:
            self.__init__(self.points, self.metric)
        # step until we get k clusters
        while self.k > k:
            self.step()
        # return list of cluster members
        return self.r, self.v


class NeuralNetwork:
    """
    Back-Propagation Neural Networks
    Placed in the public domain.
    Original author: Neil Schemenauer <nas@arctrix.com>
    Modified by: Massimo Di Pierro
    Read more: http://www.ibm.com/developerworks/library/l-neural/
    """

    @staticmethod
    def rand(a, b):
        """calculate a random number where:  a <= rand < b"""
        return (b - a) * random.random() + a

    @staticmethod
    def sigmoid(x):
        """our sigmoid function, tanh is a little nicer than the standard 1/(1+e^-x)"""
        return math.tanh(x)

    @staticmethod
    def dsigmoid(y):
        """# derivative of our sigmoid function, in terms of the output"""
        return 1.0 - y ** 2

    def __init__(self, ni, nh, no):
        # number of input, hidden, and output nodes
        self.ni = ni + 1  # +1 for bias node
        self.nh = nh
        self.no = no

        # activations for nodes
        self.ai = [1.0] * self.ni
        self.ah = [1.0] * self.nh
        self.ao = [1.0] * self.no

        # create weights
        self.wi = Matrix(self.ni, self.nh, fill=lambda r, c: self.rand(-0.2, 0.2))
        self.wo = Matrix(self.nh, self.no, fill=lambda r, c: self.rand(-2.0, 2.0))

        # last change in weights for momentum
        self.ci = Matrix(self.ni, self.nh)
        self.co = Matrix(self.nh, self.no)

    def update(self, inputs):
        if len(inputs) != self.ni - 1:
            raise ValueError("wrong number of inputs")

        # input activations
        for i in range(self.ni - 1):
            self.ai[i] = inputs[i]

        # hidden activations
        for j in range(self.nh):
            s = sum(self.ai[i] * self.wi[i, j] for i in range(self.ni))
            self.ah[j] = self.sigmoid(s)

        # output activations
        for k in range(self.no):
            s = sum(self.ah[j] * self.wo[j, k] for j in range(self.nh))
            self.ao[k] = self.sigmoid(s)
        return self.ao[:]

    def back_propagate(self, targets, N, M):
        if len(targets) != self.no:
            raise ValueError("wrong number of target values")

        # calculate error terms for output
        output_deltas = [0.0] * self.no
        for k in range(self.no):
            error = targets[k] - self.ao[k]
            output_deltas[k] = self.dsigmoid(self.ao[k]) * error

        # calculate error terms for hidden
        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh):
            error = sum(output_deltas[k] * self.wo[j, k] for k in range(self.no))
            hidden_deltas[j] = self.dsigmoid(self.ah[j]) * error

        # update output weights
        for j in range(self.nh):
            for k in range(self.no):
                change = output_deltas[k] * self.ah[j]
                self.wo[j, k] = self.wo[j, k] + N * change + M * self.co[j, k]
                self.co[j, k] = change

        # update input weights
        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j] * self.ai[i]
                self.wi[i, j] = self.wi[i, j] + N * change + M * self.ci[i, j]
                self.ci[i, j] = change

        # calculate error
        error = sum(0.5 * (targets[k] - self.ao[k]) ** 2 for k in range(len(targets)))
        return error

    def test(self, patterns):
        for p in patterns:
            print(p[0], "->", self.update(p[0]))

    def weights(self):
        print("Input weights:")
        for i in range(self.ni):
            print(self.wi[i])
        print()
        print("Output weights:")
        for j in range(self.nh):
            print(self.wo[j])

    def train(self, patterns, iterations=1000, N=0.5, M=0.1, check=False):
        # N: learning rate
        # M: momentum factor
        for i in range(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                error = error + self.back_propagate(targets, N, M)
            if check and i % 100 == 0:
                print("error %-14f" % error)


def D(f, h=1e-6):  # first derivative of f
    return lambda x, f=f, h=h: (f(x + h) - f(x - h)) / 2 / h


def DD(f, h=1e-6):  # second derivative of f
    return lambda x, f=f, h=h: (f(x + h) - 2.0 * f(x) + f(x - h)) / (h * h)


def myexp(x, precision=1e-6, max_steps=40):
    if x == 0:
        return 1.0
    elif x > 0:
        return 1.0 / myexp(-x, precision, max_steps)
    else:
        t = s = 1.0  # first term
        for k in range(1, max_steps):
            t = t * x / k  # next term
            s = s + t  # add next term
            if abs(t) < precision:
                return s
        raise ArithmeticError("no convergence")


def mysin(x, precision=1e-6, max_steps=40):
    pi = math.pi
    if x == 0:
        return 0
    elif x < 0:
        return -mysin(-x)
    elif x > 2.0 * pi:
        return mysin(x % (2.0 * pi))
    elif x > pi:
        return -mysin(2.0 * pi - x)
    elif x > pi / 2:
        return mysin(pi - x)
    elif x > pi / 4:
        return sqrt(1.0 - mysin(pi / 2 - x) ** 2)
    else:
        t = s = x  # first term
        for k in range(1, max_steps):
            t = t * (-1.0) * x * x / (2 * k) / (2 * k + 1)  # next term
            s = s + t  # add next term
            r = x ** (2 * k + 1)  # estimate residue
            if r < precision:
                return s  # stopping condition
        raise ArithmeticError("no convergence")


def mycos(x, precision=1e-6, max_steps=40):
    pi = math.pi
    if x == 0:
        return 1.0
    elif x < 0:
        return mycos(-x)
    elif x > 2.0 * pi:
        return mycos(x % (2.0 * pi))
    elif x > pi:
        return mycos(2.0 * pi - x)
    elif x > pi / 2:
        return -mycos(pi - x)
    elif x > pi / 4:
        return sqrt(1.0 - mycos(pi / 2 - x) ** 2)
    else:
        t = s = 1  # first term
        for k in range(1, max_steps):
            t = t * (-1.0) * x * x / (2 * k) / (2 * k - 1)  # next term
            s = s + t  # add next term
            r = x ** (2 * k)  # estimate residue
            if r < precision:
                return s  # stopping condition
        raise ArithmeticError("no convergence")


class Matrix(object):
    def __init__(self, rows, cols=1, fill=0.0):
        """
        Constructor a zero matrix
        Examples:
        A = Matrix([[1, 2], [3, 4]])
        A = Matrix([1, 2, 3, 4])
        A = Matrix(10, 20)
        A = Matrix(10, 20, fill=0.0)
        A = Matrix(10, 20, fill=lambda r, c: 1.0 if r == c else 0.0)
        """
        if isinstance(rows, list):
            if isinstance(rows[0], list):
                self.rows = [[e for e in row] for row in rows]
            else:
                self.rows = [[e] for e in rows]
        elif isinstance(rows, int) and isinstance(cols, int):
            xrows, xcols = range(rows), range(cols)
            if callable(fill):
                self.rows = [[fill(r, c) for c in xcols] for r in xrows]
            else:
                self.rows = [[fill for c in xcols] for r in xrows]
        else:
            raise RuntimeError("Unable to build matrix from %s" % repr(rows))
        self.nrows = len(self.rows)
        self.ncols = len(self.rows[0])

    def __getitem__(A, coords):
        "x = A[0, 1]"
        i, j = coords
        return A.rows[i][j]

    def __setitem__(A, coords, value):
        "A[0, 1] = 3.0"
        i, j = coords
        A.rows[i][j] = value

    def tolist(A):
        "assert(Matrix([[1, 2], [3, 4]]).tolist() == [[1, 2], [3, 4]])"
        return A.rows

    def __str__(A):
        return str(A.rows)

    def flatten(A):
        "assert(Matrix([[1, 2], [3, 4]]).flatten() == [1, 2, 3, 4])"
        return [A[r, c] for r in range(A.nrows) for c in range(A.ncols)]

    def reshape(A, n, m):
        "assert(Matrix([[1, 2], [3, 4]]).reshape(1, 4).tolist() == [[1, 2, 3, 4]])"
        if n * m != A.nrows * A.ncols:
            raise RuntimeError("Impossible reshape")
        flat = A.flatten()
        return Matrix(n, m, fill=lambda r, c, m=m, flat=flat: flat[r * m + c])

    def swap_rows(A, i, j):
        "assert(Matrix([[1, 2], [3, 4]]).swap_rows(1, 0).tolist() == [[3, 4], [1, 2]])"
        A.rows[i], A.rows[j] = A.rows[j], A.rows[i]

    @staticmethod
    def identity(rows=1, e=1.0):
        return Matrix(rows, rows, lambda r, c, e=e: e if r == c else 0.0)

    @staticmethod
    def diagonal(d):
        return Matrix(len(d), len(d), lambda r, c, d=d: d[r] if r == c else 0.0)

    def __add__(A, B):
        """
        Adds A and B element by element, A and B must have the same size
        Example
        >>> A = Matrix([[4, 3.0], [2, 1.0]])
        >>> B = Matrix([[1, 2.0], [3, 4.0]])
        >>> C = A + B
        >>> print(C)
        [[5, 5.0], [5, 5.0]]
        """
        n, m = A.nrows, A.ncols
        if not isinstance(B, Matrix):
            if n == m:
                B = Matrix.identity(n, B)
            elif n == 1 or m == 1:
                B = Matrix([[B for c in range(m)] for r in range(n)])
        if B.nrows != n or B.ncols != m:
            raise ArithmeticError("incompatible dimensions")
        C = Matrix(n, m)
        for r in range(n):
            for c in range(m):
                C[r, c] = A[r, c] + B[r, c]
        return C

    def __sub__(A, B):
        """
        Adds A and B element by element, A and B must have the same size
        Example
        >>> A = Matrix([[4.0, 3.0], [2.0, 1.0]])
        >>> B = Matrix([[1.0, 2.0], [3.0, 4.0]])
        >>> C = A - B
        >>> print(C)
        [[3.0, 1.0], [-1.0, -3.0]]
        """
        n, m = A.nrows, A.ncols
        if not isinstance(B, Matrix):
            if n == m:
                B = Matrix.identity(n, B)
            elif n == 1 or m == 1:
                B = Matrix(n, m, fill=B)
        if B.nrows != n or B.ncols != m:
            raise ArithmeticError("Incompatible dimensions")
        C = Matrix(n, m)
        for r in range(n):
            for c in range(m):
                C[r, c] = A[r, c] - B[r, c]
        return C

    def __radd__(A, B):  # B+A
        return A + B

    def __rsub__(A, B):  # B-A
        return (-A) + B

    def __neg__(A):
        return Matrix(A.nrows, A.ncols, fill=lambda r, c: -A[r, c])

    def __rmul__(A, x):
        "multiplies a number of matrix A by a scalar number x"
        M = copy.deepcopy(A)
        for r in range(M.nrows):
            for c in range(M.ncols):
                M[r, c] *= x
        return M

    def __mul__(A, B):
        "multiplies a number of matrix A by another matrix B"
        if isinstance(B, (list, tuple)):
            return (A * Matrix(len(B), 1, fill=lambda r, c: B[r])).nrows
        elif not isinstance(B, Matrix):
            return B * A
        elif A.ncols == 1 and B.ncols == 1 and A.nrows == B.nrows:
            # try a scalar product ;-)
            return sum(A[r, 0] * B[r, 0] for r in range(A.nrows))
        elif A.ncols != B.nrows:
            raise ArithmeticError("Incompatible dimension")
        M = Matrix(A.nrows, B.ncols)
        for r in range(A.nrows):
            for c in range(B.ncols):
                for k in range(A.ncols):
                    M[r, c] += A[r, k] * B[k, c]
        return M

    def __rtruediv__(A, x):
        """Computes x/A using Gauss-Jordan elimination where x is a scalar"""
        n = A.ncols
        if A.nrows != n:
            raise ArithmeticError("matrix not squared")
        indexes = range(n)
        A = copy.deepcopy(A)
        B = Matrix.identity(n, x)
        for c in indexes:
            for r in range(c + 1, n):
                if abs(A[r, c]) > abs(A[c, c]):
                    A.swap_rows(r, c)
                    B.swap_rows(r, c)
            p = 0.0 + A[c, c]  # trick to make sure it is not integer
            for k in indexes:
                A[c, k] = A[c, k] / p
                B[c, k] = B[c, k] / p
            for r in range(n):
                if r == c:
                    continue
                p = 0.0 + A[r, c]  # trick to make sure it is not integer
                for k in indexes:
                    A[r, k] -= A[c, k] * p
                    B[r, k] -= B[c, k] * p
        return B

    def __truediv__(A, B):
        if isinstance(B, Matrix):
            return A * (1.0 / B)  # matrix/matrix
        else:
            return (1.0 / B) * A  # matrix/scalar

    @property
    def T(A):
        """Transposed of A"""
        return Matrix(A.ncols, A.nrows, fill=lambda r, c: A[c, r])

    @property
    def H(A):
        """Hermitian of A"""
        return Matrix(A.ncols, A.nrows, fill=lambda r, c: A[c, r].conj())


def is_almost_symmetric(A, ap=1e-6, rp=1e-4):
    if A.nrows != A.ncols:
        return False
    for r in range(A.nrows):
        for c in range(r):
            delta = abs(A[r, c] - A[c, r])
            if delta > ap and delta > max(abs(A[r, c]), abs(A[c, r])) * rp:
                return False
    return True


def is_almost_zero(A, ap=1e-6, rp=1e-4):
    for r in range(A.nrows):
        for c in range(A.ncols):
            delta = abs(A[r, c] - A[c, r])
            if delta > ap and delta > max(abs(A[r, c]), abs(A[c, r])) * rp:
                return False
    return True


def norm(A, p=1):
    if isinstance(A, (list, tuple)):
        return sum(abs(x) ** p for x in A) ** (1.0 / p)
    elif isinstance(A, Matrix):
        if A.nrows == 1 or A.ncols == 1:
            return sum(
                norm(A[r, c]) ** p for r in range(A.nrows) for c in range(A.ncols)
            ) ** (1.0 / p)
        elif p == 1:
            return max(
                [sum(norm(A[r, c]) for r in range(A.nrows)) for c in range(A.ncols)]
            )
        else:
            raise NotImplementedError
    else:
        return abs(A)


def condition_number(f, x=None, h=1e-6):
    if callable(f) and not x is None:
        return D(f, h)(x) * x / f(x)
    elif isinstance(f, Matrix):  # if is the Matrix
        return norm(f) * norm(1 / f)
    else:
        raise NotImplementedError


def exp(x, ap=1e-6, rp=1e-4, ns=40):
    if isinstance(x, Matrix):
        t = s = Matrix.identity(x.ncols)
        for k in range(1, ns):
            t = t * x / k  # next term
            s = s + t  # add next term
            if norm(t) < max(ap, norm(s) * rp):
                return s
        raise ArithmeticError("no convergence")
    elif type(x) == type(1j):
        return cmath.exp(x)
    else:
        return math.exp(x)


def Cholesky(A):
    if not is_almost_symmetric(A):
        raise ArithmeticError("not symmetric")
    L = copy.deepcopy(A)
    for k in range(L.ncols):
        if L[k, k] <= 0:
            raise ArithmeticError("not positive definite")
        p = L[k, k] = math.sqrt(L[k, k])
        for i in range(k + 1, L.nrows):
            L[i, k] /= p
        for j in range(k + 1, L.nrows):
            p = float(L[j, k])
            for i in range(k + 1, L.nrows):
                L[i, j] -= p * L[i, k]
    for i in range(L.nrows):
        for j in range(i + 1, L.ncols):
            L[i, j] = 0
    return L


def is_positive_definite(A):
    if not is_almost_symmetric(A):
        return False
    try:
        Cholesky(A)
        return True
    except Exception:
        return False


def Markowitz(mu, A, r_free):
    """Assess Markowitz risk/return.
    Example:
    >>> cov = Matrix([[0.04, 0.006, 0.02],
    ...               [0.006, 0.09, 0.06],
    ...               [0.02, 0.06, 0.16]])
    >>> mu = Matrix([[0.10], [0.12], [0.15]])
    >>> r_free = 0.05
    >>> x, ret, risk = Markowitz(mu, cov, r_free)
    >>> print(x)
    [0.556634..., 0.275080..., 0.1682847...]
    >>> print(ret, risk)
    0.113915... 0.186747...
    """
    x = Matrix([[0.0] for r in range(A.nrows)])
    x = (1 / A) * (mu - r_free)
    x = x / sum(x[r, 0] for r in range(x.nrows))
    portfolio = [x[r, 0] for r in range(x.nrows)]
    portfolio_return = mu * x
    portfolio_risk = sqrt(x * (A * x))
    return portfolio, portfolio_return, portfolio_risk


def fit_least_squares(points, f):
    """
    Computes c_j for best linear fit of y[i] \pm dy[i] = fitting_f(x[i])
    where fitting_f(x[i]) is \sum_j c_j f[j](x[i])

    parameters:
    - a list of fitting functions
    - a list with points (x, y, dy)

    returns:
    - column vector with fitting coefficients
    - the chi2 for the fit
    - the fitting function as a lambda x: ....
    """

    def eval_fitting_function(f, c, x):
        if len(f) == 1:
            return c * f[0](x)
        else:
            return sum(func(x) * c[i, 0] for i, func in enumerate(f))

    A = Matrix(len(points), len(f))
    b = Matrix(len(points))
    for i in range(A.nrows):
        weight = 1.0 / points[i][2] if len(points[i]) > 2 else 1.0
        b[i, 0] = weight * float(points[i][1])
        for j in range(A.ncols):
            A[i, j] = weight * f[j](float(points[i][0]))
    c = (1.0 / (A.T * A)) * (A.T * b)
    chi = A * c - b
    chi2 = norm(chi, 2) ** 2
    fitting_f = lambda x, c=c, f=f, q=eval_fitting_function: q(f, c, x)
    cs = [c] if isinstance(c, float) else c.flatten()
    return cs, chi2, fitting_f


# examples of fitting functions
def POLYNOMIAL(n):
    return [(lambda x, p=p: x ** p) for p in range(n + 1)]


CONSTANT = POLYNOMIAL(0)
LINEAR = POLYNOMIAL(1)
QUADRATIC = POLYNOMIAL(2)
CUBIC = POLYNOMIAL(3)
QUARTIC = POLYNOMIAL(4)


class Trader:
    def model(self, window):
        "the forecasting model"
        # we fit last few days quadratically
        points = list(enumerate(window))
        a, chi2, fitting_f = fit_least_squares(points, QUADRATIC)
        # and we extrapolate tomorrow"s price
        tomorrow_prediction = fitting_f(len(points))
        return tomorrow_prediction

    def strategy(self, history, ndays=7):
        "the trading strategy"
        if len(history) < ndays:
            return
        else:
            today_close = history[-1]
            tomorrow_prediction = self.model(history[-ndays:])
            return "buy" if tomorrow_prediction > today_close else "sell"

    def simulate(self, data, cash=1000.0, shares=0.0, daily_rate=0.03 / 360):
        "find fitting parameters that optimize the trading strategy"
        for t in range(len(data)):
            suggestion = self.strategy(data[:t])
            today_close = data[t - 1]
            # and we buy or sell based on our strategy
            if cash > 0 and suggestion == "buy":
                # we keep track of finances
                shares_bought = int(cash / today_close)
                shares += shares_bought
                cash -= shares_bought * today_close
            elif shares > 0 and suggestion == "sell":
                cash += shares * today_close
                shares = 0.0
            # we assume money in the bank also gains an interest
            cash *= math.exp(daily_rate)
        # we return the net worth
        return cash + shares * data[-1]


def fake_stock_prices(start_price=100, average_return=0.05, volatility=0.30, days=100):
    daily_volatility = volatility / math.sqrt(250)
    daily_return = average_return / 250
    s = [random.gauss(0, daily_volatility) for d in range(days - 1)]
    mu = daily_return - sum(s) / (days - 1)
    v = [start_price]
    for item in s:
        v.append(v[-1] * math.exp(mu + item))
    return v


def sqrt(x):
    try:
        return math.sqrt(x)
    except ValueError:
        return cmath.sqrt(x)


def Jacobi_eigenvalues(A, checkpoint=False):
    """Returns U end e so that A=U * diagonal(e) * transposed(U)
    where i-column of U contains the eigenvector corresponding to
    the eigenvalue e[i] of A.

    from http://en.wikipedia.org/wiki/Jacobi_eigenvalue_algorithm
    """

    def maxind(M, k):
        j = k + 1
        for i in range(k + 2, M.ncols):
            if abs(M[k, i]) > abs(M[k, j]):
                j = i
        return j

    n = A.nrows
    if n != A.ncols:
        raise ArithmeticError("matrix not squared")
    indexes = range(n)
    S = Matrix(n, n, fill=lambda r, c: float(A[r, c]))
    E = Matrix.identity(n)
    state = n
    ind = [maxind(S, k) for k in indexes]
    e = [S[k, k] for k in indexes]
    changed = [True for k in indexes]
    iteration = 0
    while state:
        if checkpoint:
            checkpoint("rotating vectors (%i) ..." % iteration)
        m = 0
        for k in range(1, n - 1):
            if abs(S[k, ind[k]]) > abs(S[m, ind[m]]):
                m = k
            pass
        k, h = m, ind[m]
        p = S[k, h]
        y = (e[h] - e[k]) / 2
        t = abs(y) + sqrt(p * p + y * y)
        s = sqrt(p * p + t * t)
        c = t / s
        s = p / s
        t = p * p / t
        if y < 0:
            s, t = -s, -t
        S[k, h] = 0
        y = e[k]
        e[k] = y - t
        if changed[k] and y == e[k]:
            changed[k], state = False, state - 1
        elif (not changed[k]) and y != e[k]:
            changed[k], state = True, state + 1
        y = e[h]
        e[h] = y + t
        if changed[h] and y == e[h]:
            changed[h], state = False, state - 1
        elif (not changed[h]) and y != e[h]:
            changed[h], state = True, state + 1
        for i in range(k):
            S[i, k], S[i, h] = c * S[i, k] - s * S[i, h], s * S[i, k] + c * S[i, h]
        for i in range(k + 1, h):
            S[k, i], S[i, h] = c * S[k, i] - s * S[i, h], s * S[k, i] + c * S[i, h]
        for i in range(h + 1, n):
            S[k, i], S[h, i] = c * S[k, i] - s * S[h, i], s * S[k, i] + c * S[h, i]
        for i in indexes:
            E[k, i], E[h, i] = c * E[k, i] - s * E[h, i], s * E[k, i] + c * E[h, i]
        ind[k], ind[h] = maxind(S, k), maxind(S, h)
        iteration += 1
    # sort vectors
    for i in range(1, n):
        j = i
        while j > 0 and e[j - 1] > e[j]:
            e[j], e[j - 1] = e[j - 1], e[j]
            E.swap_rows(j, j - 1)
            j -= 1
    # normalize vectors
    U = Matrix(n, n)
    for i in indexes:
        norm = sqrt(sum(E[i, j] ** 2 for j in indexes))
        for j in indexes:
            U[j, i] = E[i, j] / norm
    return U, e


def compute_correlation_matrix(v):
    """Input is a list of lists where some elements can be missing (None)"""
    m = len(v)
    n = len(v[0])
    mus = [sum(v[i][k] for k in range(n)) / n for i in range(m)]
    var = [sum(v[i][k] ** 2 for k in range(n)) / n - mus[i] ** 2 for i in range(m)]
    corr = Matrix(
        m,
        m,
        fill=lambda i, j: (
            sum(v[i][k] * v[j][k] for k in range(1, n)) / n - mus[i] * mus[j]
        )
        / math.sqrt(var[i] * var[j]),
    )
    return corr


def invert_minimum_residual(f, x, ap=1e-4, rp=1e-4, ns=200):
    y = copy.copy(x)
    r = x - 1.0 * f(x)
    for k in range(ns):
        q = f(r)
        alpha = (q * r) / (q * q)
        y = y + alpha * r
        r = r - alpha * q
        residue = sqrt((r * r) / r.nrows)
        if residue < max(ap, norm(y) * rp):
            return y
    raise ArithmeticError("no convergence")


def invert_bicgstab(f, x, ap=1e-4, rp=1e-4, ns=200):
    y = copy.copy(x)
    r = x - 1.0 * f(x)
    q = r
    p = 0.0
    s = 0.0
    rho_old = alpha = omega = 1.0
    for k in range(ns):
        rho = q * r
        beta = (rho / rho_old) * (alpha / omega)
        rho_old = rho
        p = beta * p + r - (beta * omega) * s
        s = f(p)
        alpha = rho / (q * s)
        r = r - alpha * s
        t = f(r)
        omega = (t * r) / (t * t)
        y = y + omega * r + alpha * p
        residue = sqrt((r * r) / r.nrows)
        if residue < max(ap, norm(y) * rp):
            return y
        r = r - omega * t
    raise ArithmeticError("no convergence")


def solve_fixed_point(f, x, ap=1e-6, rp=1e-4, ns=100):
    def g(x):
        return f(x) + x  # f(x)=0 <=> g(x)=x

    Dg = D(g)
    for k in range(ns):
        if abs(Dg(x)) >= 1:
            raise ArithmeticError("error D(g)(x)>=1")
        (x_old, x) = (x, g(x))
        if k > 2 and norm(x_old - x) < max(ap, norm(x) * rp):
            return x
    raise ArithmeticError("no convergence")


def solve_bisection(f, a, b, ap=1e-6, rp=1e-4, ns=100):
    fa, fb = f(a), f(b)
    if fa == 0:
        return a
    if fb == 0:
        return b
    if fa * fb > 0:
        raise ArithmeticError("f(a) and f(b) must have opposite sign")
    for k in range(ns):
        x = (a + b) / 2
        fx = f(x)
        if fx == 0 or norm(b - a) < max(ap, norm(x) * rp):
            return x
        elif fx * fa < 0:
            (b, fb) = (x, fx)
        else:
            (a, fa) = (x, fx)
    raise ArithmeticError("no convergence")


def solve_newton(f, x, ap=1e-6, rp=1e-4, ns=20):
    x = float(x)  # make sure it is not int
    for k in range(ns):
        (fx, Dfx) = (f(x), D(f)(x))
        if norm(Dfx) < ap:
            raise ArithmeticError("unstable solution")
        (x_old, x) = (x, x - fx / Dfx)
        if k > 2 and norm(x - x_old) < max(ap, norm(x) * rp):
            return x
    raise ArithmeticError("no convergence")


def solve_secant(f, x, ap=1e-6, rp=1e-4, ns=20):
    x = float(x)  # make sure it is not int
    (fx, Dfx) = (f(x), D(f)(x))
    for k in range(ns):
        if norm(Dfx) < ap:
            raise ArithmeticError("unstable solution")
        (x_old, fx_old, x) = (x, fx, x - fx / Dfx)
        if k > 2 and norm(x - x_old) < max(ap, norm(x) * rp):
            return x
        fx = f(x)
        Dfx = (fx - fx_old) / (x - x_old)
    raise ArithmeticError("no convergence")


def optimize_bisection(f, a, b, ap=1e-6, rp=1e-4, ns=100):
    return solve_bisection(D(f), a, b, ap, rp, ns)


def optimize_newton(f, x, ap=1e-6, rp=1e-4, ns=20):
    x = float(x)  # make sure it is not int
    (f, Df) = (D(f), DD(f))
    for k in range(ns):
        (fx, Dfx) = (f(x), Df(x))
        if Dfx == 0:
            return x
        if norm(Dfx) < ap:
            raise ArithmeticError("unstable solution")
        (x_old, x) = (x, x - fx / Dfx)
        if norm(x - x_old) < max(ap, norm(x) * rp):
            return x
    raise ArithmeticError("no convergence")


def optimize_secant(f, x, ap=1e-6, rp=1e-4, ns=100):
    x = float(x)  # make sure it is not int
    (f, Df) = (D(f), DD(f))
    (fx, Dfx) = (f(x), Df(x))
    for k in range(ns):
        if fx == 0:
            return x
        if norm(Dfx) < ap:
            raise ArithmeticError("unstable solution")
        (x_old, fx_old, x) = (x, fx, x - fx / Dfx)
        if norm(x - x_old) < max(ap, norm(x) * rp):
            return x
        fx = f(x)
        Dfx = (fx - fx_old) / (x - x_old)
    raise ArithmeticError("no convergence")


def optimize_golden_search(f, a, b, ap=1e-6, rp=1e-4, ns=100):
    a, b = float(a), float(b)
    tau = (sqrt(5.0) - 1.0) / 2.0
    x1, x2 = a + (1.0 - tau) * (b - a), a + tau * (b - a)
    fa, f1, f2, fb = f(a), f(x1), f(x2), f(b)
    for k in range(ns):
        if f1 > f2:
            a, fa, x1, f1 = x1, f1, x2, f2
            x2 = a + tau * (b - a)
            f2 = f(x2)
        else:
            b, fb, x2, f2 = x2, f2, x1, f1
            x1 = a + (1.0 - tau) * (b - a)
            f1 = f(x1)
        if k > 2 and norm(b - a) < max(ap, norm(b) * rp):
            return b
    raise ArithmeticError("no convergence")


def partial(f, i, h=1e-4):
    def df(x, f=f, i=i, h=h):
        x = list(x)  # make copy of x
        x[i] += h
        f_plus = f(x)
        x[i] -= 2 * h
        f_minus = f(x)
        if isinstance(f_plus, (list, tuple)):
            return [(f_plus[i] - f_minus[i]) / (2 * h) for i in range(len(f_plus))]
        else:
            return (f_plus - f_minus) / (2 * h)

    return df


def gradient(f, x, h=1e-4):
    return Matrix(len(x), 1, fill=lambda r, c: partial(f, r, h)(x))


def hessian(f, x, h=1e-4):
    return Matrix(len(x), len(x), fill=lambda r, c: partial(partial(f, r, h), c, h)(x))


def jacobian(f, x, h=1e-4):
    partials = [partial(f, c, h)(x) for c in range(len(x))]
    return Matrix(len(partials[0]), len(x), fill=lambda r, c: partials[c][r])


def solve_newton_multi(f, x, ap=1e-6, rp=1e-4, ns=20):
    """
    Computes the root of a multidimensional function f near point x.

    Parameters
    f is a function that takes a list and returns a scalar
    x is a list

    Returns x, solution of f(x)=0, as a list
    """
    n = len(x)
    x = Matrix(len(x))
    for k in range(ns):
        fx = Matrix(f(x.flatten()))
        J = jacobian(f, x.flatten())
        if norm(J) < ap:
            raise ArithmeticError("unstable solution")
        (x_old, x) = (x, x - (1.0 / J) * fx)
        if k > 2 and norm(x - x_old) < max(ap, norm(x) * rp):
            return x.flatten()
    raise ArithmeticError("no convergence")


def optimize_newton_multi(f, x, ap=1e-6, rp=1e-4, ns=20):
    """
    Finds the extreme of multidimensional function f near point x.

    Parameters
    f is a function that takes a list and returns a scalar
    x is a list

    Returns x, which maximizes of minimizes f(x)=0, as a list
    """
    x = Matrix(list(x))
    for k in range(ns):
        (grad, H) = (gradient(f, x.flatten()), hessian(f, x.flatten()))
        if norm(H) < ap:
            raise ArithmeticError("unstable solution")
        (x_old, x) = (x, x - (1.0 / H) * grad)
        if k > 2 and norm(x - x_old) < max(ap, norm(x) * rp):
            return x.flatten()
    raise ArithmeticError("no convergence")


def optimize_newton_multi_imporved(f, x, ap=1e-6, rp=1e-4, ns=20, h=10.0):
    """
    Finds the extreme of multidimensional function f near point x.

    Parameters
    f is a function that takes a list and returns a scalar
    x is a list

    Returns x, which maximizes of minimizes f(x)=0, as a list
    """
    x = Matrix(list(x))
    fx = f(x.flatten())
    for k in range(ns):
        (grad, H) = (gradient(f, x.flatten()), hessian(f, x.flatten()))
        if norm(H) < ap:
            raise ArithmeticError("unstable solution")
        (fx_old, x_old, x) = (fx, x, x - (1.0 / H) * grad)
        fx = f(x.flatten())
        while fx > fx_old:  # revert to steepest descent
            (fx, x) = (fx_old, x_old)
            norm_grad = norm(grad)
            (x_old, x) = (x, x - grad / norm_grad * h)
            (fx_old, fx) = (fx, f(x.flatten()))
            h = h / 2
        h = norm(x - x_old) * 2
        if k > 2 and h / 2 < max(ap, norm(x) * rp):
            return x.flatten()
    raise ArithmeticError("no convergence")


def fit(data, fs, b=None, ap=1e-6, rp=1e-4, ns=200, constraint=None):
    if not isinstance(fs, (list, tuple)):

        def g(b, data=data, f=fs, constraint=constraint):
            chi2 = sum(((y - f(b, x)) / dy) ** 2 for (x, y, dy) in data)
            if constraint:
                chi2 += constraint(b)
            return chi2

        if isinstance(b, (list, tuple)):
            b = optimize_newton_multi_imporved(g, b, ap, rp, ns)
        else:
            b = optimize_newton(g, b, ap, rp, ns)
        return b, g(b, data, constraint=None)
    elif not b:
        a, chi2, ff = fit_least_squares(data, fs)
        return a, chi2
    else:
        na = len(fs)

        def core(b, data=data, fs=fs):
            A = Matrix([[fs[k](b, x) / dy for k in range(na)] for (x, y, dy) in data])
            z = Matrix([[y / dy] for (x, y, dy) in data])
            a = (1 / (A.T * A)) * (A.T * z)
            chi2 = norm(A * a - z) ** 2
            return a.flatten(), chi2

        def g(b, data=data, fs=fs, constraint=constraint):
            a, chi2 = core(b, data, fs)
            if constraint:
                chi += constraint(b)
            return chi2

        b = optimize_newton_multi_imporved(g, b, ap, rp, ns)
        a, chi2 = core(b, data, fs)
        return a + b, chi2


def integrate_naive(f, a, b, n=20):
    """
    Integrates function, f, from a to b using the trapezoidal rule
    >>> from math import sin
    >>> integrate(sin, 0, 2)
    1.416118...
    """
    a, b = float(a), float(b)
    h = (b - a) / n
    return h / 2 * (f(a) + f(b)) + h * sum(f(a + h * i) for i in range(1, n))


def integrate(f, a, b, ap=1e-4, rp=1e-4, ns=20):
    """
    Integrates function, f, from a to b using the trapezoidal rule
    converges to precision
    """
    I = integrate_naive(f, a, b, 1)
    for k in range(1, ns):
        I_old, I = I, integrate_naive(f, a, b, 2 ** k)
        if k > 2 and norm(I - I_old) < max(ap, norm(I) * rp):
            return I
    raise ArithmeticError("no convergence")


class QuadratureIntegrator:
    """
    Calculates the integral of the function f from points a to b
    using n Vandermonde weights and numerical quadrature.
    """

    def __init__(self, order=4):
        h = 1.0 / (order - 1)
        A = Matrix(order, order, fill=lambda r, c: (c * h) ** r)
        s = Matrix(order, 1, fill=lambda r, c: 1.0 / (r + 1))
        w = (1 / A) * s
        self.w = w

    def integrate(self, f, a, b):
        w = self.w
        order = len(w.rows)
        h = float(b - a) / (order - 1)
        return (b - a) * sum(w[i, 0] * f(a + i * h) for i in range(order))


def integrate_quadrature_naive(f, a, b, n=20, order=4):
    a, b = float(a), float(b)
    h = float(b - a) / n
    q = QuadratureIntegrator(order=order)
    return sum(q.integrate(f, a + i * h, a + i * h + h) for i in range(n))


def E(f, S):
    return float(sum(f(x) for x in S)) / (len(S) or 1)


def mean(X):
    return E(lambda x: x, X)


def variance(X):
    return E(lambda x: x ** 2, X) - E(lambda x: x, X) ** 2


def sd(X):
    return sqrt(variance(X))


def covariance(X, Y):
    return sum(X[i] * Y[i] for i in range(len(X))) / len(X) - mean(X) * mean(Y)


def correlation(X, Y):
    return covariance(X, Y) / sd(X) / sd(Y)


class MCG(object):
    def __init__(self, seed, a=66539, m=2 ** 31):
        self.x = seed
        self.a, self.m = a, m

    def next(self):
        self.x = (self.a * self.x) % self.m
        return self.x

    def random(self):
        return float(self.next()) / self.m


class MarsenneTwister(object):
    """
    based on:
    Knuth 1981, The Art of Computer Programming
    Vol. 2 (2nd Ed.), pp102]
    """

    def __init__(self, seed=4357):
        self.w = []  # the array for the state vector
        self.w.append(seed & 0xFFFFFFFF)
        for i in range(1, 625):
            self.w.append((69069 * self.w[i - 1]) & 0xFFFFFFFF)
        self.wi = i

    def random(self):
        w = self.w
        wi = self.wi
        N, M, U, L = 624, 397, 0x80000000, 0x7FFFFFFF
        K = [0x0, 0x9908B0DF]
        y = 0
        if wi >= N:
            for kk in range((N - M) + 1):
                y = (w[kk] & U) | (w[kk + 1] & L)
                w[kk] = w[kk + M] ^ (y >> 1) ^ K[y & 0x1]

            for kk in range(kk, N):
                y = (w[kk] & U) | (w[kk + 1] & L)
                w[kk] = w[kk + (M - N)] ^ (y >> 1) ^ K[y & 0x1]
            y = (w[N - 1] & U) | (w[0] & L)
            w[N - 1] = w[M - 1] ^ (y >> 1) ^ K[y & 0x1]
        wi = 0
        y = w[wi]
        wi += 1
        y ^= y >> 11
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        y ^= y >> 18
        return float(y) / 0xFFFFFFFF


def leapfrog(mcg, k):
    a = mcg.a ** k % mcg.m
    return [MCG(mcg.next(), a, mcg.m) for i in range(k)]


class RandomSource(object):
    def __init__(self, generator=None):
        self.generator = generator or random

    def random(self):
        return self.generator.random()

    def randint(self, a, b):
        return int(a + (b - a + 1) * self.random())

    def choice(self, S):
        return S[self.randint(0, len(S) - 1)]

    def bernoulli(self, p):
        return 1 if self.random() < p else 0

    def lookup(self, table, epsilon=1e-6):
        if isinstance(table, dict):
            table = table.items()
        u = self.random()
        for key, p in table:
            if u < p + epsilon:
                return key
            u = u - p
        raise ArithmeticError("invalid probability")

    def binomial(self, n, p, epsilon=1e-6):
        u = self.random()
        q = (1.0 - p) ** n
        for k in range(n + 1):
            if u < q + epsilon:
                return k
            u = u - q
            q = q * (n - k) / (k + 1) * p / (1.0 - p)
        raise ArithmeticError("invalid probability")

    def negative_binomial(self, k, p, epsilon=1e-6):
        u = self.random()
        n = k
        q = p ** k
        while True:
            if u < q + epsilon:
                return n
            u = u - q
            q = q * n / (n - k + 1) * (1 - p)
            n = n + 1
        raise ArithmeticError("invalid probability")

    def poisson(self, lamb, epsilon=1e-6):
        u = self.random()
        q = exp(-lamb)
        k = 0
        while True:
            if u < q + epsilon:
                return k
            u = u - q
            q = q * lamb / (k + 1)
            k = k + 1
        raise ArithmeticError("invalid probability")

    def uniform(self, a, b):
        return a + (b - a) * self.random()

    def exponential(self, lamb):
        return -log(self.random()) / lamb

    def gauss(self, mu=0.0, sigma=1.0):
        if hasattr(self, "other") and self.other:
            this, other = self.other, None
        else:
            while True:
                v1 = self.random(-1, 1)
                v2 = self.random(-1, 1)
                r = v1 * v1 + v2 * v2
                if r < 1:
                    break
            this = sqrt(-2.0 * log(r) / r) * v1
            self.other = sqrt(-2.0 * log(r) / r) * v1
        return mu + sigma * this


def confidence_intervals(mu, sigma):
    """Computes the normal confidence intervals"""
    CONFIDENCE = [
        (0.68, 1.0),
        (0.80, 1.281551565545),
        (0.90, 1.644853626951),
        (0.95, 1.959963984540),
        (0.98, 2.326347874041),
        (0.99, 2.575829303549),
        (0.995, 2.807033768344),
        (0.998, 3.090232306168),
        (0.999, 3.290526731492),
        (0.9999, 3.890591886413),
        (0.99999, 4.417173413469),
    ]
    return [(a, mu - b * sigma, mu + b * sigma) for (a, b) in CONFIDENCE]

    def pareto(self, alpha, xm):
        u = self.random()
        return xm * (1.0 - u) ** (-1.0 / alpha)

    def point_on_circle(self, radius=1.0):
        angle = 2.0 * pi * self.random()
        return radius * math.cos(angle), radius * math.sin(angle)

    def point_in_circle(self, radius=1.0):
        while True:
            x = self.uniform(-radius, radius)
            y = self.uniform(-radius, radius)
            if x * x + y * y < radius * radius:
                return x, y

    def point_in_sphere(self, radius=1.0):
        while True:
            x = self.uniform(-radius, radius)
            y = self.uniform(-radius, radius)
            z = self.uniform(-radius, radius)
            if x * x + y * y * z * z < radius * radius:
                return x, y, z

    def point_on_sphere(self, radius=1.0):
        x, y, z = self.point_in_sphere(radius)
        norm = math.sqrt(x * x + y * y + z * z)
        return x / norm, y / norm, z / norm


def resample(S, size=None):
    return [random.choice(S) for i in range(size or len(S))]


def bootstrap(x, confidence=0.68, nsamples=100):
    """Computes the bootstrap errors of the input list."""

    def mean(S):
        return float(sum(x for x in S)) / len(S)

    means = [mean(resample(x)) for k in range(nsamples)]
    means.sort()
    left_tail = int(((1.0 - confidence) / 2) * nsamples)
    right_tail = nsamples - 1 - left_tail
    return means[left_tail], mean(x), means[right_tail]


class MCEngine:
    """
    Monte Carlo Engine parent class.
    Runs a simulation many times and computes average and error in average.
    Must be extended to provide the simulate_once method
    """

    def simulate_once(self):
        raise NotImplementedError

    def simulate_many(self, ap=0.1, rp=0.1, ns=1000):
        self.results = []
        s1 = s2 = 0.0
        self.convergence = False
        for k in range(1, ns):
            x = self.simulate_once()
            self.results.append(x)
            s1 += x
            s2 += x * x
            mu = float(s1) / k
            variance = float(s2) / k - mu * mu
            dmu = sqrt(variance / k)
            if k > 10:
                if abs(dmu) < max(ap, abs(mu) * rp):
                    self.convergence = True
                    break
        self.results.sort()
        return bootstrap(self.results)

    def var(self, confidence=95):
        index = int(0.01 * len(self.results) * confidence + 0.999)
        if len(self.results) - index < 5:
            raise ArithmeticError("not enough data, not reliable")
        return self.results[index]
