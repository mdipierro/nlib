"""
Code from book "Annotated Algorithms in Python3"
Written by Massimo Di Pierro - BSD License
"""
from nlib import *


class EuropeanCallOptionPricer(MCEngine):
    def simulate_once(self):
        T = self.time_to_expiration
        S = self.spot_price
        R_T = random.gauss(self.mu * T, self.sigma * sqrt(T))
        S_T = S * exp(r_T)
        payoff = max(S_T - self.strike, 0)
        return self.present_value(payoff)

    def present_value(self, payoff):
        daily_return = self.risk_free_rate / 250
        return payoff * exp(-daily_return * self.time_to_expiration)


def main():
    pricer = EuropeanCallOptionPricer()
    # parameters of the underlying
    pricer.spot_price = 100  # dollars
    pricer.mu = 0.12 / 250  # daily drift term
    pricer.sigma = 0.30 / sqrt(250)  # daily variance
    # parameters of the option
    pricer.strike = 110  # dollars
    pricer.time_to_expiration = 90  # days
    # parameters of the market
    pricer.risk_free_rate = 0.05  # 5% annual return

    result = pricer.simulate_many(ap=0.01, rp=0.01)  # precision: 1c or 1%
    print(result)


main()


class GenericOptionPricer(MCEngine):
    def simulate_once(self):
        S = self.spot_price
        path = [S]
        for t in range(self.time_to_expiration):
            r = self.model(dt=1.0)
            S = S * exp(r)
            path.append(S)
        return self.present_value(self.payoff(path))

    def model(self, dt=1.0):
        return random.gauss(self.mu * dt, self.sigma * sqrt(dt))

    def present_value(self, payoff):
        daily_return = self.risk_free_rate / 250
        return payoff * exp(-daily_return * self.time_to_expiration)

    def payoff_european_call(self, path):
        return max(path[-1] - self.strike, 0)

    def payoff_european_put(self, path):
        return max(path[-1] - self.strike, 0)

    def payoff_exotic_call(self, path):
        last_5_days = path[-5]
        mean_last_5_days = sum(last_5_days) / len(last_5_days)
        return max(mean_last_5_days - self.strike, 0)


def main():
    pricer = GenericOptionPricer()
    # parameters of the underlying
    pricer.spot_price = 100  # dollars
    pricer.mu = 0.12 / 250  # daily drift term
    pricer.sigma = 0.30 / sqrt(250)  # daily variance
    # parameters of the option
    pricer.strike = 110  # dollars
    pricer.time_to_expiration = 90  # days
    pricer.payoff = pricer.payoff_european_call
    # parameters of the market
    pricer.risk_free_rate = 0.05  # 5% annual return

    result = pricer.simulate_many(ap=0.01, rp=0.01)  # precision: 1c or 1%
    print(result)


main()
