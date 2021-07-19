"""
Code from book "Annotated Algorithms in Python3"
Written by Massimo Di Pierro - BSD License
"""
from random import choice, randint


class Chromosome:
    alphabet = "ATGC"
    size = 32
    mutations = 2

    def __init__(self, father=None, mother=None):
        if not father or not mother:
            self.dna = [choice(self.alphabet) for i in range(self.size)]
        else:
            self.dna = father.dna[: self.size / 2] + mother.dna[self.size / 2 :]
            for mutation in range(self.mutations):
                self.dna[randint(0, self.size - 1)] = choice(self.alphabet)

    def fitness(self, target):
        return sum(1 for i, c in enumerate(self.dna) if c == target.dna[i])


def top(population, target, n=10):
    table = [(chromo.fitness(target), chromo) for chromo in population]
    table.sort(reverse=True)
    return [row[1] for row in table][:n]


def oneof(population):
    return population[randint(0, len(population) - 1)]


def main():
    GENERATIONS = 10000
    OFFSPRING = 20
    SEEDS = 20
    TARGET = Chromosome()

    population = [Chromosome() for i in range(SEEDS)]
    for i in range(GENERATIONS):
        print("\n\nGENERATION:", i)
        print(0, TARGET.dna)
        fittest = top(population, TARGET)
        for chromosome in fittest:
            print(i, chromosome.dna)
        if max(chromo.fitness(TARGET) for chromo in fittest) == Chromosome.size:
            print("SOLUTION FOUND")
            break
        population = [
            Chromosome(father=oneof(fittest), mother=oneof(fittest))
            for i in range(OFFSPRING)
        ]


if __name__ == "__main__":
    main()
