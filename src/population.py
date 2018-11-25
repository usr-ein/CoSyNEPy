import numpy as np
from neural_network import NeuralNetwork
import helpers

class Population():
    """docstring for Population"""
    def __init__(self, n, m):
        self.n = n
        self.m = m

        self.P = np.random.rand(n, m).astype(np.float32)
        self.fitnesses = np.random.rand(n, m).astype(np.float32)
        self.marked = np.zeros((self.n, self.m))

        self.currentGeneration = 0

    def updateFitnesses(self, newFitnesses):
        self.fitnesses *= self.currentGeneration # Undo the mean
        self.currentGeneration += 1 # Updates the update count

        #assert newFitnesses.shape[0] == m == self.fitnesses.shape[0]
        #assert newFitnesses.shape[1] == 0
        self.fitnesses += newFitnesses[np.newaxis, :] # Add each mean to corresponding columm
        
        self.fitnesses /= self.currentGeneration # Redo the mean

    def sort(self, i):
        newIndices = np.argsort(-self.fitnesses[i, :])
        self.P[i] = self.P[i, newIndices]
        self.fitnesses[i] = self.fitnesses[i, newIndices]

    def replaceLastGenes(self, i, l, offsprings):
        self.P[i, -l:] = offsprings.P[i,:]
        self.fitnesses[i, -l:] = offsprings.fitnesses[i,:]

    def probability(self, i):
        minFit = self.fitnesses[i].min()
        maxFit = self.fitnesses[i].max()
        if maxFit == minFit:
            return np.array([0.5]*self.m)

        probabilities = 1 - np.power((self.fitnesses[i] - minFit) / (maxFit - minFit), 1 / self.n)
        return probabilities

    def mark(self, i, j):
        self.marked[i, j] = 1

    def resetMarked(self):
        self.marked = np.zeros((self.n, self.m))

    def permuteMarked(self, i):
        # Row telling us what to do
        rowMarkers      = self.marked[i,:]
        count2Permutate = int(rowMarkers.sum())
        if count2Permutate <= 1:
            return
        # Row to be permutated
        P_i             = self.P[i,:]
        # Row that will be acted upon
        P_i_permuted    = P_i.copy() # Necessary, again
        # Pairs of genes to permutate in the row i
        pairs = helpers.random_derangement(count2Permutate)
        indicesOfNotNull = np.where(rowMarkers)[0]
        for p1, p2 in zip(indicesOfNotNull, indicesOfNotNull[pairs]):
            # p1 is the starting index to permute
            # p2 is the final index where p1 is going
            P_i_permuted[p1], P_i_permuted[p2] = P_i[p2], P_i[p1]
        self.P[i,:] = P_i_permuted

    def buildNetwork(self, j, psi):
        X = self.P[:,j]
        matricesElementCount = np.multiply(psi[:-1], psi[1:])
        splitIndices = np.cumsum(matricesElementCount)[:-1]
        M = np.split(X, splitIndices)
        weightMatrices = []
        for weights, layerIndex in zip(M, range(1, len(psi))):
            weightMatrices.append(weights.reshape(psi[layerIndex-1], psi[layerIndex]))
            
        weightMatrices = np.array(weightMatrices)

        return NeuralNetwork(weightMatrices=weightMatrices, psi=psi)


