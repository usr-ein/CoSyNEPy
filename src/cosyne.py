import numpy as np
import random
from population import Population
from recombinator import Recombinator

class CoSyNE():
    """docstring for CoSyNE"""
    def __init__(self, m, psi, evaluator, topRatioToRecombine=0.25, ratioToMutate=0.10, maxFitness=None):
        self.m = m
        self.psi = psi
        self.maxFitness = maxFitness
        self.evaluator = evaluator

        self.topRatioToRecombine = topRatioToRecombine
        self.ratioToMutate = ratioToMutate
        self.maxFitness = maxFitness

        self.n, self.n_biases, self.l = self.computeOtherParameters()

        self.population = Population(self.n, self.m, self.n_biases)
        self.recombinator = Recombinator(self.topRatioToRecombine, self.ratioToMutate)

    def computeOtherParameters(self):
        # number of weights
        n = sum(self.psi[i - 1] * self.psi[i] for i in range(1, len(self.psi)))
        # number rof weights + biases
        n_biases = len(self.psi) - 1
        l = int(self.m * self.topRatioToRecombine)
        return n, n_biases, l

    def evolve(self):
        fitnesses = self.evaluator.evaluatePop(self.population, self.psi)

        self.population.updateFitnesses(fitnesses)

        offsprings = self.recombinator.recombine(self.population)

        for i in range(self.n): # does not act on biases
            self.population.sort(i)

            self.population.replaceLastGenes(i, self.l, offsprings)

            probabilities = self.population.probability(i)

            for j in range(self.m):
                if random.random() < probabilities[j]:
                    self.population.mark(i,j)
            self.population.permuteMarked(i)

        self.population.resetMarked()

        if self.maxFitness:
            if fitnesses.mean() > self.maxFitness:
                return True
            return False

