"""
Code from book "Annotated Algorithms in Python3"
Written by Massimo Di Pierro - BSD License
"""
import math
import random

from nlib import Canvas, mean, sd


class Ising:
    def __init__(self, n):
        self.n = n
        self.s = [[[1 for x in range(n)] for y in range(n)] for z in range(n)]
        self.magnetization = n ** 3

    def __getitem__(self, point):
        n = self.n
        x, y, z = point
        return self.s[(x + n) % n][(y + n) % n][(z + n) % n]

    def __setitem__(self, point, value):
        n = self.n
        x, y, z = point
        self.s[(x + n) % n][(y + n) % n][(z + n) % n] = value

    def step(self, t, h):
        n = self.n
        x, y, z = (
            random.randint(0, n - 1),
            random.randint(0, n - 1),
            random.randint(0, n - 1),
        )
        neighbors = [
            (x - 1, y, z),
            (x + 1, y, z),
            (x, y - 1, z),
            (x, y + 1, z),
            (x, y, z - 1),
            (x, y, z + 1),
        ]
        dE = (
            -2.0
            * self[x, y, z]
            * (h + sum(self[xn, yn, zn] for xn, yn, zn in neighbors))
        )
        if dE > t * math.log(random.random()):
            self[x, y, z] = -self[x, y, z]
            self.magnetization += 2 * self[x, y, z]
        return self.magnetization


def simulate(steps=100):
    ising = Ising(n=10)
    data = {}
    for h in range(0, 11):  # external magnetic field
        data[h] = []
        for t in range(1, 11):  # temperature, in units of K
            m = [ising.step(t=t, h=h) for k in range(steps)]
            mu = mean(m)  # average magnetization
            sigma = sd(m)
            data[h].append((t, mu, sigma))
    return data


def main(name="ising.png"):
    data = simulate(steps=10000)
    canvas = Canvas(xlab="temperature", ylab="magnetization")
    for h in data:
        color = "#%.2x0000" % (h * 25)
        canvas.errorbar(data[h]).plot(data[h], color=color)
    canvas.save(name)


main()
