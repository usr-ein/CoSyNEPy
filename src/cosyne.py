from random import random as fastRandom
import numpy as np
# Fast sigmoid, slower for single value cause C overhead
from scipy.special import expit
from scipy.optimize import rosen as rosenSciPy # The Rosenbrock function

def main():
    trainer = CoSyNE(20, [1,3,1], topRatioToRecombine=0.25, ratioToMutate=0.20, verbose=4)
    for e in range(3):
        trainer.evolve()

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
    while same.size > 0:  # while not empty
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

    def __init__(self, weightMatrices, activationFunctions=None, costFunction=None):
        '''Initialise the NeuralNetwork.
        Parameters
        ----------
        weightMatrices : np.ndarray
            Array of weight matrix. Each matrix represent the transition from Layer_i to Layer_i+1,
            hence the next matrix shall have the same number of rows as the number of columns of the previous.
            This coherence can and will be tested through the checkCoherenceWeights method.
        activationFunctions : list[functions]
            List of the activation functions for each layer, default is relu until last layer, then sigmoid.
        costFunction : function
            Fuction to compute cost, should be like cost(pred, targets) --> float32. Default: RMS error
        '''
        # Array of matrices of size Layer_i x Layer_i+1
        # where lines contains weights from nth neurone in Layer_i to all the m neurones in Layer_i+1
        self.weightMatrices = weightMatrices
        self.depth = weightMatrices.shape[0]
        # Array of the activation functions to use, must be of size self.depth
        if activationFunctions == None:
            self.activationFunctions = [self.relu] * \
                (self.depth-1) + [self.sigmoid]
        else:
            self.activationFunctions = activationFunctions

        if costFunction == None:
            self.costFunction = self.rmse
        else:
            self.costFunction = costFunction
        self.checkCoherenceWeights()  # Verifies that all the weights are well defined


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

class CoSyNE():
    '''Cooperative Synapse NeuroEvolution trainer'''

    def __init__(self, m, psi, topRatioToRecombine=0.25, ratioToMutate=0.20, verbose=1):
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

        self.markedForPermutation = self.initMarkedForPermutation()

        self.currentGeneration = 0

        self.verbose = verbose
        if self.verbose > 0:
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

    def prob(self, P, coords):
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
        n = P.shape[0]
        fit = P[i, j, 1]
        minFit = P[i, :, 1].min()
        maxFit = P[i, :, 1].max()
        return 1 - np.power((fit - minFit) / (maxFit - minFit), 1 / n)

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

        assert psi[0] == psi[-1] # For the rosenbrock
        inputs = np.random.rand(psi[0])
        targets = rosenSciPy(inputs)

        predictions = network.forward(inputs)
        cost = network.costFunction(predictions, targets)
        if self.verbose > 3:
            print("Cost {}", cost)

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
        if self.verbose > 3:
            print("Permutating {} marked genes at random".format(count2Permutate))
        # Pairs of genes to permutate in the row i
        pairs = random_derangement(count2Permutate)
        indicesOfNotNull = np.where(rowMarkers)[0]
        for p1, p2 in zip(indicesOfNotNull, indicesOfNotNull[pairs]):
            # p1 is the starting index to permute
            # p2 is the final index where p1 is going
            P_i_permuted[p1], P_i_permuted[p2] = P_i[p2], P_i[p1]
        self.P[i,:] = P_i_permuted


    def evolve(self):

        if self.verbose > 0:
            print("Generation {} starts".format(self.currentGeneration))

        for j in range(self.m):
            X = self.P[:, j]
            X_fitness = self.evaluate(X[:,0], self.psi)
            self.updateGenesFitness(X[:,1], X_fitness, self.currentGeneration)

        # Sort P in place
        self.sortSubpopulations(self.P)

        if self.verbose > 0:
            print("Top fitness after evaluation : {}".format(self.P[:,0,1].mean()))

        if self.verbose > 2:
            print("Recombining populations...")
        # Crossover then mutates
        O = self.recombine(
            self.P,
            ratioToMutate=self.ratioToMutate,
            topRatioToRecombine=self.topRatioToRecombine)

        for i in range(self.n):

            # l is equivalent to countToRecombine
            l = int(self.m * self.topRatioToRecombine)

            if self.verbose > 2:
                print("Replacing the {} least fit genes of population {}...".format(l, i))
            for k in range(l):
                self.P[i, self.m - k - 1] = O[i, k]

            if self.verbose > 2:
                print("Selecting genes in population {} to be permutated...".format(i))
            for j in range(self.m):
                if fastRandom() < self.prob(self.P, (i, j)):
                    self.mark((i, j))
            if self.verbose > 2:
                print("Permutating population {}...".format(i))
            self.permuteMarked(i)

        self.markedForPermutation = self.initMarkedForPermutation()
        self.currentGeneration += 1

if __name__ == '__main__':
    main()