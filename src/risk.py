"""
Code from book "Annotated Algorithms in Python3"
Written by Massimo Di Pierro - BSD License
"""
import random

from nlib import *


class RiskEngine(MCEngine):
    def __init__(self, lamb, xm, alpha):
        self.lamb = lamb
        self.xm = xm
        self.alpha = alpha

    def simulate_once(self):
        total_loss = 0.0
        t = 0.0
        while t < 260:
            dt = random.expovariate(self.lamb)
            amount = self.xm * random.paretovariate(self.alpha)
            t += dt
            total_loss += amount
        return total_loss


def main():
    s = RiskEngine(lamb=10, xm=5000, alpha=1.5)
    print(s.simulate_many(rp=1e-4, ns=1000))
    print(s.var(95))


main()
