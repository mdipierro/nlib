"""
Code from book "Annotated Algorithms in Python3"
Written by Massimo Di Pierro - BSD License
"""


class MCIntegrator(MCEngine):
    def __init__(self, f, a, b):
        self.f = f
        self.a = a
        self.b = b

    def simulate_once(self):
        a, b, f = self.a, self.b, self.f
        x = a + (b - a) * random.random()
        g = (b - a) * f(x)
        return g


def main():
    s = MCIntegrator(f=lambda x: math.sin(x), a=0, b=1)
    print(s.simulate_many())


main()
