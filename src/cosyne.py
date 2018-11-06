from random import random as fastRandom
import numpy as np
# Fast sigmoid, slower for single value cause C overhead
from scipy.special import expit

# Random permutations without fixed points a.k.a. derangement
# About 40% slower than np.random.permutation but that's not much
# 59.4 µs ± 4.13 µs
# See https://stackoverflow.com/a/52614780/5989906


def random_derangement(n):
    ''' Random permutations without fixed points a.k.a. derangement.
    Parameters
    ----------
        n : int
            Size of the range of integers to find the random derangement of.

    Notes
    -----
    About 40% slower than np.random.permutation but that's not much
    59.4 µs ± 4.13 µs
    See https://stackoverflow.com/a/52614780/5989906
    '''
    original = np.arange(n)
    new = np.random.permutation(n)
    same = np.where(original == new)[0]
    while same:  # while not empty
        swap = same[np.random.permutation(len(same))]
        new[same] = new[swap]
        same = np.where(original == new)[0]
        if len(same) == 1:
            swap = np.random.randint(0, n)
            new[[same[0], swap]] = new[[swap, same[0]]]
    return new


def normalTrucatedMultiple(n, size=1):
    ''' Return size samples of the normal distribution around
    n/2 truncated so that min is 0 and max is n-1.

    Parameters
    ----------
    n : int
        Parameter of the distribution according to desc.
    size: int, optional
        Output shape.
        If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn.
        If size is None (default), a single value is returned if loc and scale are both scalars.
        Otherwise, np.broadcast(loc, scale).size samples are drawn.

    Notes
    -----
    For one sample:
    3.8 µs ± 839 ns
    But for size=5000 samples:
    129 µs ± 1.05 µs
    '''

    return np.random.normal(
        n * 0.5, n * 0.33, size=size).astype(int).clip(0, n - 1)


class NeuralNetwork():
    '''Representation of a classical feed forward multi layer perceptron.

    This representation is shape agnostic, meaning that each layer can have a
    different shape as long as they're all fully connected (i.e. that the weight matrix is
    coherent).
    '''

    def __init__(self, weightMatrices):
        '''Initialise the NeuralNetwork.
        Parameters
        ----------
        weightMatrices : np.ndarray
            Array of weight matrix. Each matrix represent the transition from Layer_i to Layer_i+1,
            hence the next matrix shall have the same number of rows as the number of columns of the previous.
            This coherence can and will be tested through the checkCoherenceWeights method.
        '''
        # Array of matrices of size Layer_i x Layer_i+1
        # where lines contains weights from nth neurone in Layer_i to all the m neurones in Layer_i+1
        self.weightMatrices = weightMatrices
        self.depth = weightMatrices.shape[0]
        # Array of the activation functions to use, must be of size self.depth
        # TODO: Parametrize this list
        self.activationFunctions = [self.relu] * \
            (self.depth-1) + [self.sigmoid]
        checkCoherenceWeights(
        )  # Verifies that all the weights are well defined

    def checkCoherenceWeights(self):
        previousOutput = self.weightMatrices[0].shape[1]
        for weightMatrix in self.weightMatrices[1:]:
            s = weightMatrix.shape
            assert s[0] == previousOutput
            previousOutput = s[1]

    def forward(self, x):
        layer = x
        for weightMatrix, activationFunction in zip(self.weightMatrices,
                                                    self.activationFunctions):
            layer = np.dot(layer, weightMatrix)
            layer = activationFunction(layer)
        return layer

    def sigmoid(self, x):
        return expit(x)

    def relu(self, x):
        x[x < 0] = 0
        return x

    def softmax(self, x):
        '''Compute softmax values for each sets of scores in x.'''
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def elu(self, x, a=2):
        '''exponential linear unit, from this paper  https://arxiv.org/abs/1511.07289... seems to work quite well'''
        return np.where(x <= 0, a * (np.exp(x) - 1), x)

    def gaussian(self, x):
        return np.exp(np.negative(np.square(x)))

    def rmse(self, pred, targets):
        return np.sqrt(((pred - targets)**2).mean()).astype('float32')


class CoSyNE(object):
    '''Cooperative Synapse NeuroEvolution trainer'''

    def __init__(self, m, psi, topRatioToRecombine=0.25, ratioToMutate=0.20):
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
        self.P = np.random.rand(self.n, self.m, 2)

        self.markedForPermutation = np.zeros(self.n, self.m)

    def recombine(P_sorted, ratioToMutate, topRatioToRecombine=0.25):
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
            n, m = O.shape

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
            n, m = O.shape

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

    def prob(P, coords):
        ''' Computes the probability concluding if the gene will be switch in its row.

        Parameters
        ----------
        P : ndarray
            Array of shape (n, m, 2) where rows are populations of genes,
            columns are complete genotypes, first depth the weight value and
            second depth its fitness.
        coords : (int, int)
            Indexes (row, col) of the gene to check the probability of.
        '''
        i, j = coords
        fit = P[i, j, 1]
        minFit = P[i, :, 1].min(axis=1)
        maxFit = P[i, :, 1].max(axis=1)
        return 1 - np.power((fit - minFit) / (maxFit - minFit), 1 / n)

    def sortSubpopulations(P):
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

    def evaluate(X, psi):
        # TODO: implement OpenAI's Gym here
        pass

    ###### TODO : TEST mark AND perrnuteMarked !!!! ######
    def mark(coords):
        i, j = coords
        self.markedForPermutation[i,j] = 1

    def permuteMarked(P, i):
        # Pairs of genes to permutate in the row i
        pairs = random_derangement(self.markedForPermutation[i,:].sum())
        indicesOfNotNull = np.where(self.markedForPermutation)[0]
        for p1, p2 in zip(indicesOfNotNull, indicesOfNotNull[pairs]):
            # p1 is the starting index to permute
            # p2 is the final index where p1 is going
            P[i, p1], P[i, p2] = P[i, p2], P[i, p1].copy()



    def evolve(self):
        for j in range(self.m):
            X = self.P[:, j]
            self.P[:, j] = evaluate(X, self.psi)

        # Sort P in place
        sortSubpopulations(self.P)
        # Crossover then mutates
        O = recombine(
            self.P,
            ratioToMutate=self.ratioToMutate,
            topRatioToRecombine=self.topRatioToRecombine)

        for i in range(self.n):
            # l is equivalent to countToRecombine
            l = int(self.m * self.topRatioToRecombine)

            for k in range(l):
                self.P[i, self.m - k] = O[i, k]

            for j in range(self.m):
                if fastRandom() < self.prob(i, j):
                    mark((i, j))
            P[i,:] = permuteMarked(P, i)