from random import random as fastRandom
import numpy as np
#from numba import jit
from scipy.optimize import rosen as rosenSciPy # The Rosenbrock function

# From this project
from neural_network import NeuralNetwork
from helpers import random_derangement, normalTrucatedMultiple

class CoSyNE():
    '''Cooperative Synapse NeuroEvolution trainer'''

    def __init__(self, m, psi, topRatioToRecombine=0.25, ratioToMutate=0.20, verbose=True):
        '''Initialise the Cooperative Synapse NeuroEvolution trainer.

        The n parameter is not necessary as it will be deduced from psi.

        Parameters
        ----------
        m : int
            Number of possible complete solution to work with and evaluate. The more there are, the more diversity
            hence a best solution can be found, but it converges slower. See figure 2 page 5.
        psi : np.ndarray
            Array of neurone count on each layer. For instance if the networks has 3 inputs, 1 hidden layer with 5 neurones
            and one hidden layer with 3 then 2 outputs, psi = np.array([3, 5, 3, 2]) . Use numpy array even if the list
            seems small, we won't modify psi so let's optimise it.
        topRatioToRecombine : float, optional
            Ratio of the least fitted population to replace with offsprings genotypes, between 0 and 1.
            Also, ratio of the best fitted genes in the population to recombine.
        ratioToMutate : float, optional
            Ratio of the offspring population to mutate randomly, between 0 and 1.
        '''
        self.m = m
        self.psi = psi
        self.topRatioToRecombine = topRatioToRecombine
        self.ratioToMutate = ratioToMutate
        # Counts the number of weights required to run the psi network architecture
        self.n = sum(psi[i - 1] * psi[i] for i in range(1, len(psi)))
        # Rows are sub-populations, columns are complete genotypes, second depth is fitness
        self.P = np.random.rand(self.n, self.m, 2).astype(np.float32)

        self.markedForPermutation = self.initMarkedForPermutation()

        self.currentGeneration = 0


        # Below parameters are not essential to the working of CoSyNE
        self.lastBestFitness = 0
        self.lastImprovedGen = 0

        self.verbose = verbose
        if self.verbose:
            print("Initialising CoSyNE\n", '#'*40)
            print("Number of genotypes to evolve: ", self.m)
            print("Network architecture (psi)   : ", self.psi)
            print("Top ratio to recombined      : ", self.topRatioToRecombine)
            print("Ratio of children to mutate  : ", self.ratioToMutate)
            print("Number of weights to be evolved per genotypes: ", self.n)
            print("Population matrix shape:     : ", self.P.shape)
            print('#'*40)

    def recombine(self, P_sorted, ratioToMutate, topRatioToRecombine=0.25):
        ''' Recombines the top-quarter complete genotypes. 
        According to the paper: 
        "After all of the networks have been evaluated and assigned a fitness, 
        the **top quarter** (a.k.a. topRatioToRecombine=0.25) with the highest fitness (i.e., the parents) are recombined
        using crossover and mutation.""

        Parameters
        ----------
        P_sorted : ndarray
            Array of shape (n, m, 2) where lines are populations,
            cols are complete genotypes, first depth are the weights
            and second depth are their fitness.
        ratioToMutate : float
            Ratio between 0 and 1 of the child population O to mutate
        topRatioToRecombine : float, optional
            Ratio between 0 and 1 of the parent population to recombine.
            The paper states "the top quarter", so default is 0.25 (quarter).
        '''

        # Crossing over
        # This crossover is in-place on O, there will always be
        # one gene that is crossovered twice for each recombination
        def crossover(O):
            ''' Crossover at a random point of the parents' chromosome.
            Acts in-place on the given array.

            Parameters
            ----------
            O : ndarray
                Array of shape (n, m,) to perform the crossover of its m columns onto.
            '''
            # Those n, m are different from self.n and self.m
            n, m, _ = O.shape

            pairs = random_derangement(m)
            crossIndices = normalTrucatedMultiple(n, m)
            for p1, p2 in zip(range(m), pairs):
                x = crossIndices[p1]
                # this copy is necessary
                O[:, p1][:x], O[:, p2][:x] = O[:, p2][:x], O[:, p1][:x].copy()

        def mutate(O, ratioToMutate):
            ''' Mutates at random point(s) of the chromosomes.
            Acts in-place on the given array.

            Parameters
            ----------
            O : ndarray
                Array of shape (n, m,) to perform the mutations onto.
            ratioToMutate : float
                Ratio between 0 and 1 of the population O to mutate
            '''
            # Those n, m are different from self.n and self.m
            n, m, _ = O.shape

            countToMutate = int(n * m * ratioToMutate)
            # Random indices allowing for repeatition (i.e. replace=True)
            random_i = np.random.choice(n, countToMutate, replace=True)
            random_j = np.random.choice(m, countToMutate, replace=True)
            for i, j in zip(random_i, random_j):
                O[i, j, 0] = fastRandom()  # Random float between 0 an 1

        countToRecombine = int(self.m * topRatioToRecombine)
        # top of the pop will be recombined
        O = P_sorted[:, :countToRecombine, :].copy()
        # Crossover between random combination of O's col (complete genotypes)
        crossover(O)
        # Mutates randomly percentage of the child population O
        mutate(O, ratioToMutate=ratioToMutate)

        return O

    def probability(self, fitness, minFit, maxFit):
        ''' Computes the probability concluding if the gene will be switch in its row.

        Parameters
        ----------
        fitness : float32
            Fitness of the gene of which to calculate the probability
        minFit : float32
            Minimum fitness in the gene's population
        maxFit : float32
            Minimum fitness in the gene's population

        Returns
        -------
        float32
            The probability (between 0 and 1) to switch the genes
        '''
        if minFit == maxFit or fit == minFit:
            return 1
        return 1 - np.power((fitness - minFit) / (maxFit - minFit), 1 / self.n)

    def sortSubpopulations(self, P):
        ''' Sorts each column (i.e. complete genotype) by the avg fitness of its values (genes)
            in descending order (best first).
            The population ndarray must be of shape (n, m, 2) where the second depth is fitness.
            Returns the sorted ndarray.
            Doesn't act in-place.

            Parameters
            ----------
            P : ndarray
                Array of shape (n, m, 2) to perform the sorting onto.

            Returns
            -------
            ndarray
                The column's fitness-wise sorted matrix.
        '''
        # Calc the mean of the fitness of each column (a.k.a. complete genotype)
        meansFitnessGenotypes = np.mean(P[:, :, 1], axis=0)
        # Reorder P in desc order according to the fitness of each column (meansFitnessGenotypes)
        return P[:, np.argsort(-meansFitnessGenotypes), :]

    def constructWeightMatrices(self, X, psi):
        ''' Constructs the required weight matrices that contains the weight to use
            to propagate the neural network forward. Returns them as a list of ndarray.

            Parameters
            ----------
            X : ndarray
                Array of shape (m,) to construct the weight matrices from.
            psi : np.ndarray
                Array of neurone count on each layer. For instance if the networks has 3 inputs, 1 hidden layer with 5 neurones
                and one hidden layer with 3 then 2 outputs, psi = np.array([3, 5, 3, 2]) . Use numpy array even if the list
                seems small, we won't modify psi so let's optimise it.

            Returns
            -------
            ndarray
                List of weight matrices of required shapes to run the fully connected psi network.
        '''

        # Counts the number of values needed to construct each weight matrix of the network
        matricesElementCount = lambda psi : [psi[i - 1] * psi[i] for i in range(1, len(psi))]
        splitIndices = np.cumsum(matricesElementCount(psi))[:-1]
        # Splits the X complete genotypes into the required number of weights matrices,
        # with the good number of values inside them, but in shape of a vector
        M = np.split(X, splitIndices)

        # Reshapes the vectors in M into matrices with the right sizes
        weightMatrices = []
        for weights, layerIndex in zip(M, range(1, len(psi))):
            weightMatrices.append(weights.reshape(psi[layerIndex-1], psi[layerIndex]))
            
        return np.array(weightMatrices)

    def evaluate(self, X, psi):
        # TODO  long term: implement OpenAI's Gym here
        weightMatrices = self.constructWeightMatrices(X, psi)

        network = NeuralNetwork(weightMatrices=weightMatrices)

        # assert psi[0] == psi[-1] # For the rosenbrock
        inputs = np.random.rand(psi[0])
        targets = rosenSciPy(inputs)

        predictions = network.forward(inputs)
        cost = network.costFunction(predictions, targets)

        return cost

    def updateGenesFitness(self, X, X_evalFitness, currentGeneration):
        ''' Recalculate the fitness of each weight in the tested genotype based on their previous
            average fitness and the newly tested fitness.
            Acts in-place on the X vector provided which should be of shape (n, 1).

            Parameters
            ----------
            X : np.ndarray
                The second depth of the genotype column, i.e. P[:, j, 1], or even P[:, j][:, 1]. Will be updated in-place.
            X_evalFitness : float32
                The fitness of the network generated from that same X genotype, evaluated with the evaluate(X, psi) method.
            currentGeneration: int
                The current generation count, starts at 0

            Notes
            -----
            The formula is from https://math.stackexchange.com/a/22351 but is trivial.
        '''
        X = (X_evalFitness-X)/(currentGeneration+1) + X

    def mark(self, coords):
        i, j = coords
        self.markedForPermutation[i,j] = 1

    def initMarkedForPermutation(self):
        return np.zeros((self.n, self.m))

    def permuteMarked(self, i):
        ''' Permutates genes in population i which position is marked as 1 in
            self.markedForPermutation.
            Acts in-place on the self.P[i,:] population array.

            Parameters
            ----------
            i : int
                Index of the row in P to be permutated.

        '''
        # Row to be permutated
        P_i             = self.P[i,:]
        # Row that will be acted upon
        P_i_permuted    = P_i.copy()
        # Row telling us what to do
        rowMarkers      = self.markedForPermutation[i,:]
        count2Permutate = int(rowMarkers.sum())
        if count2Permutate <= 1:
            return
        # Pairs of genes to permutate in the row i
        pairs = random_derangement(count2Permutate)
        indicesOfNotNull = np.where(rowMarkers)[0]
        for p1, p2 in zip(indicesOfNotNull, indicesOfNotNull[pairs]):
            # p1 is the starting index to permute
            # p2 is the final index where p1 is going
            P_i_permuted[p1], P_i_permuted[p2] = P_i[p2], P_i[p1]
        self.P[i,:] = P_i_permuted


    def evolve(self):
        for j in range(self.m):
            X = self.P[:, j]
            X_fitness = self.evaluate(X[:,0], self.psi)
            self.updateGenesFitness(X[:,1], X_fitness, self.currentGeneration)

        # Sort P in place
        self.P = self.sortSubpopulations(self.P)

        currentBestFitness = self.P[:,0,1].mean()
        if self.lastBestFitness == currentBestFitness:
            countLastImproved = self.currentGeneration - self.lastImprovedGen
            if self.currentGeneration > 3 and countLastImproved > self.lastImprovedGen:
                exit()
        else:
            self.lastBestFitness = currentBestFitness
            self.lastImprovedGen = self.currentGeneration

        if self.verbose:
            currentBestFitness = self.P[:,0,1].mean()
            if self.lastBestFitness != currentBestFitness:
                print(" "*30, end='\r')
                print("Generation {}\t|\t".format(self.currentGeneration), end='')
                print("Top fitness increased : {}".format(currentBestFitness))
                self.lastBestFitness = currentBestFitness
                self.lastImprovedGen = self.currentGeneration
            else:
                countLastImproved = self.currentGeneration - self.lastImprovedGen
                print("==> {}/{} generations since last improvement".format(countLastImproved, self.lastImprovedGen), end="\r")
                if self.currentGeneration > 3 and countLastImproved > self.lastImprovedGen:
                    print("More than {} generations since last improvement, stopping..".format(self.lastImprovedGen))
                    exit()

        # Crossover then mutates
        O = self.recombine(
            self.P,
            ratioToMutate=self.ratioToMutate,
            topRatioToRecombine=self.topRatioToRecombine)

        # l is equivalent to countToRecombine
        l = int(self.m * self.topRatioToRecombine)

        for i in range(self.n):

            # Calculate this once per population as it's expensive
            minFit = self.P[i, :, 1].min()
            maxFit = self.P[i, :, 1].max()

            for k in range(l):
                self.P[i, self.m - k - 1] = O[i, k]

            for j in range(self.m):
                if fastRandom() < self.probability(fitness=self.P[i, j, 1], minFit=minFit, maxFit=minFit):
                    self.mark((i, j))
            self.permuteMarked(i)

        self.markedForPermutation = self.initMarkedForPermutation()
        self.currentGeneration += 1