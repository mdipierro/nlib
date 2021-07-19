"""
Code from book "Annotated Algorithms in Python3"
Written by Massimo Di Pierro - BSD License
"""
import math
import random

from nlib import *


class NuclearReactor(MCEngine):
    def __init__(self, radius, mean_free_path=1.91, threshold=200):
        self.radius = radius
        self.density = 1.0 / mean_free_path
        self.threshold = threshold

    def point_on_sphere(self):
        while True:
            x, y, z = random.random(), random.random(), random.random()
            d = math.sqrt(x * x + y * y + z * z)
            if d < 1:
                return (x / d, y / d, z / d)  # project on surface

    def simulate_once(self):
        p = (0, 0, 0)
        events = [p]
        while events:
            event = events.pop()
            v = self.point_on_sphere()
            d1 = random.expovariate(self.density)
            d2 = random.expovariate(self.density)
            p1 = (p[0] + v[0] * d1, p[1] + v[1] * d1, p[2] + v[2] * d1)
            p2 = (p[0] - v[0] * d2, p[1] - v[1] * d2, p[2] - v[2] * d2)
            if p1[0] ** 2 + p1[1] ** 2 + p1[2] ** 2 < self.radius:
                events.append(p1)
            if p2[0] ** 2 + p2[1] ** 2 + p2[2] ** 2 < self.radius:
                events.append(p2)
            if len(events) > self.threshold:
                return 1.0
        return 0.0


def main():
    s = NuclearReactor(MCEngine)
    data = []
    s.radius = 0.01
    while s.radius < 21:
        r = s.simulate_many(ap=0.01, rp=0.01, ns=1000, nm=100)
        data.append((s.radius, r[1], (r[2] - r[0]) / 2))
        s.radius *= 1.2
    c = Canvas(title="Critical Mass", xlab="Radius", ylab="Probability Chain Reaction")
    c.plot(data).errorbar(data).save("nuclear.png")


main()
