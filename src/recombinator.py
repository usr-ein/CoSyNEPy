import numpy as np
import random
from copy import copy

from helpers import random_derangement, normalTrucatedMultiple
from population import Population

class Recombinator():
    """docstring for Recombinator"""
    def __init__(self, topRatioToRecombine, ratioToMutate):
        self.topRatioToRecombine = topRatioToRecombine
        self.ratioToMutate = ratioToMutate

    def crossover(self, O):
        pairs = random_derangement(O.m)
        crossIndices = normalTrucatedMultiple(O.n, O.m)
        for p1, p2 in zip(range(O.m), pairs):
            x = crossIndices[p1]
            # this copy is necessary
            # same lines but different array
            O.P[:, p1][:x], O.P[:, p2][:x] = O.P[:, p2][:x], O.P[:, p1][:x].copy()
            O.fitnesses[:, p1][:x], O.fitnesses[:, p2][:x] = O.fitnesses[:, p2][:x], O.fitnesses[:, p1][:x].copy()

    def mutate(self, O):
        # Mutations on weights
        countToMutate = int(O.n * O.m * self.ratioToMutate)
        # Random indices allowing for repeatition (i.e. replace=True)
        random_i = np.random.choice(O.n, countToMutate, replace=True)
        random_j = np.random.choice(O.m, countToMutate, replace=True)
        for i, j in zip(random_i, random_j):
            O.P[i, j] = random.random()
            O.fitnesses[i, j] = random.random()

        # Mutations on biases
        countToMutate = int(O.n_biases * O.m * self.ratioToMutate)
        random_i = np.random.choice(O.n_biases, countToMutate, replace=True)
        random_j = np.random.choice(O.m, countToMutate, replace=True)
        for i, j in zip(random_i, random_j):
            O.biases[i, j] = random.random()

    def sortColByMeanFitness(self, population):
        newPopulation = Population(population.n, population.m, population.n_biases)
        meansFitnessGenotypes = np.mean(population.fitnesses, axis=0)
        # Reorder P in desc order according to the fitness of each column (meansFitnessGenotypes)
        newOrder = np.argsort(-meansFitnessGenotypes)
        newPopulation.P = population.P[:, newOrder]
        newPopulation.fitnesses = population.fitnesses[:, newOrder]
        newPopulation.biases = population.biases[:, newOrder]
        return newPopulation

    def recombine(self, population):
        countToRecombine = int(population.m * self.topRatioToRecombine)
        popSorted = self.sortColByMeanFitness(population)
        O = Population(population.n, countToRecombine, population.n_biases)
        O.P = popSorted.P[:, :countToRecombine].copy()
        O.fitnesses = popSorted.fitnesses[:, :countToRecombine].copy()
        O.biases = popSorted.biases[:, :countToRecombine].copy()
        self.crossover(O)
        self.mutate(O)

        return O