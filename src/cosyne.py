import random
import numpy as np
import h5py

# From this project
from neural_network import NeuralNetwork
from helpers import random_derangement, normalTrucatedMultiple

class CoSyNE():
    '''Cooperative Synapse NeuroEvolution trainer'''

    def __init__(self, m=None, psi=None, evaluator=None, costFunction=None, activationFunctions=None, topRatioToRecombine=0.25, ratioToMutate=0.20, seed=None, verbose=True, loadFile=None):
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
        seed : int, optional
            Seed to initialise the pseudo random number generator
        verbose : Bool
            Whether or not to display stuff
        '''

        if seed != None:
            np.random.seed(seed)
            random.seed(seed)

        self.verbose = verbose

        if m is None or psi is None or evaluator is None:
            if loadFile is None:
                raise ValueError("At least m and psi should be provided, or a save file")
            else:
                self.importCurrentGeneration(loadFile)
                if self.verbose:
                    self.displayParameters()
                return

        self.m = m
        self.psi = psi
        self.evaluator = evaluator
        self.costFunction = costFunction
        self.activationFunctions = activationFunctions
        self.topRatioToRecombine = topRatioToRecombine
        self.ratioToMutate = ratioToMutate

        # l is equivalent to countToRecombine
        self.l = int(self.m * self.topRatioToRecombine)
        # Counts the number of weights required to run the psi network architecture
        self.n = sum(psi[i - 1] * psi[i] for i in range(1, len(psi)))
        # Rows are sub-populations, columns are complete genotypes, second depth is fitness
        # second depth is random but will be discarded when updated for the first time
        self.P = np.random.rand(self.n, self.m, 2).astype(np.float32)

        self.markedForPermutation = self.initMarkedForPermutation()

        self.currentGeneration = 0
        self.STOP = False # When set to true, evolve() doesn't do anything anymore

        # Below parameters are not essential to the working of CoSyNE
        self.lastImprovedGen = 0
        self.bestFitnessPerGen = []

        if self.verbose:
            self.displayParameters()

    def displayParameters(self):
        print("Initialising CoSyNE\n", '#'*40)
        print("Number of genotypes to evolve: ", self.m)
        print("Network architecture (psi)   : ", self.psi)
        print("Top ratio to recombined      : ", self.topRatioToRecombine)
        print("Ratio of children to mutate  : ", self.ratioToMutate)
        print("Number of weights to be evolved per genotypes: ", self.n)
        print("Population matrix shape:     : ", self.P.shape)
        print('#'*40)

    def recombine(self):
        ''' Recombines the top-quarter complete genotypes. 
        According to the paper: 
        "After all of the networks have been evaluated and assigned a fitness, 
        the **top quarter** (a.k.a. topRatioToRecombine=0.25) with the highest fitness (i.e., the parents) are recombined
        using crossover and mutation.""
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
                O[i, j, 0] = random.random()  # Random float between 0 an 1

        countToRecombine = int(self.m * self.topRatioToRecombine)

        P_sorted = self.sortGenotypesByMeanFitness(inplace=False)
        # top of the pop will be recombined
        O = P_sorted[:, :countToRecombine, :].copy()
        # Crossover between random combination of O's col (complete genotypes)
        crossover(O)
        # Mutates randomly percentage of the child population O
        mutate(O, ratioToMutate=self.ratioToMutate)

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

    def sortGenotypesByMeanFitness(self, inplace=True):
        ''' Sorts each column (i.e. complete genotype) by the avg fitness of its values (genes)
            in descending order (best first).
            The population ndarray must be of shape (n, m, 2) where the second depth is fitness.
        '''
        # Calc the mean of the fitness of each column (a.k.a. complete genotype)
        meansFitnessGenotypes = np.mean(self.P[:, :, 1], axis=0)
        # Reorder P in desc order according to the fitness of each column (meansFitnessGenotypes)
        sortedP = self.P[:, np.argsort(-meansFitnessGenotypes), :]
        if inplace:
            self.P = sortedP
        else:
            return sortedP

    def sortSubpopulationGenesByFitness(self, i, inplace=True):
        ''' Sorts each gene (i.e. P[i,j]) inside the subpopulation P[i, :] by their fitness
            in desc order (best first).
            Returns the sorted ndarray.

            Parameters
            ----------
            i : int
                Index of the population to perform the sorting onto.
        '''

        sortedP_i = self.P[i, np.argsort(-self.P[i, :, 1]), :]
        if inplace:
            self.P[i] = sortedP_i
        else:
            return sortedP_i

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
        matricesElementCount = np.multiply(psi[:-1], psi[1:])
        splitIndices = np.cumsum(matricesElementCount)[:-1]
        # Splits the X complete genotypes into the required number of weights matrices,
        # with the good number of values inside them, but in shape of a vector
        M = np.split(X, splitIndices)

        # Reshapes the vectors in M into matrices with the right sizes
        weightMatrices = []
        for weights, layerIndex in zip(M, range(1, len(psi))):
            weightMatrices.append(weights.reshape(psi[layerIndex-1], psi[layerIndex]))
            
        return np.array(weightMatrices)

    def constructNetwork(self, X, psi):
        ''' Constructs the neural network based on the given genotype and network architecture.
            Parameters
            ----------
            X : ndarray
                Array of shape (m,) to construct the weight matrices from.
            psi : np.ndarray
                Array of neurone count on each layer. For instance if the networks has 3 inputs, 1 hidden layer with 5 neurones
                and one hidden layer with 3 then 2 outputs, psi = np.array([3, 5, 3, 2]) . Use numpy array even if the list
                seems small, we won't modify psi so let's optimise it.
        '''
        weightMatrices = self.constructWeightMatrices(X, psi)
        return NeuralNetwork(weightMatrices=weightMatrices, psi=self.psi, activationFunctions=self.activationFunctions, costFunction=self.costFunction)

    def evaluate(self, X, psi):
        network = self.constructNetwork(X, psi)
        fitness = self.evaluator(network)
        return fitness

    def updateGenesFitness(self, j, evalFitness):
        ''' Recalculate the fitness of each weight in the tested genotype based on their previous
            average fitness and the newly tested fitness.
            Acts in-place on the X vector provided which should be of shape (n, 1).

            Parameters
            ----------
            j : np.ndarray
                Index of the genotype column to be updated.
            evalFitness : float32
                The fitness of the network generated from that same X genotype, evaluated with the evaluate(X, psi) method.

            Notes
            -----
            The formula is from https://math.stackexchange.com/a/22351 but is trivial.
        '''
        #X = (X_evalFitness-X)/(currentGeneration+1) + X
        self.P[:,j,1] = (self.P[:,j,1] * self.currentGeneration + evalFitness)/(self.currentGeneration+1)

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

    def exportBestNetwork(self, outputFile):
        ''' Exports the current best network into an HDF5 file with datasets 'weightMatrices/n'
            for the n-th weight matrix, and 'networkArchitecture' containing the psi architecture.

            Parameters
            ----------
            outputFile : string
                Path and filename where to save the HDF5 file.

        '''
        bestGenotype = self.P[:,0,0] 
        weightMatrices = self.constructWeightMatrices(bestGenotype, self.psi)
        with h5py.File(outputFile, 'w') as hf:
            for i in range(len(weightMatrices)):
                hf.create_dataset("weightMatrices/{}".format(i), data=weightMatrices[i])
            hf.create_dataset("networkArchitecture", data=self.psi)

    def exportCurrentGeneration(self, outputFile):
        ''' Exports the current state of the trainer into an HDF5 file with datasets 'populationMatrix',
            'currentGeneration', 'networkArchitecture', 'topRatioToRecombine', 'ratioToMutate'
            corresponding to the parameters with the same name (except networkArchitecture that corresponds to psi).

            Parameters
            ----------
            outputFile : string
                Path and filename where to save the HDF5 file.

        '''
        with h5py.File(outputFile, 'w') as hf:
            hf.create_dataset("populationMatrix", data=self.P)
            hf.create_dataset("networkArchitecture", data=self.psi)
            hf.create_dataset("metaInfo", data=np.array([self.currentGeneration, self.topRatioToRecombine, self.ratioToMutate]))
            hf.create_dataset("bestFitnessPerGen", data=np.array(self.bestFitnessPerGen))

    def importCurrentGeneration(self, inputFile):
        ''' Imports the current state of the trainer that was saved in an HDF5 file.
            Recalculates the necessary informations from it as well.

            Parameters
            ----------
            inputFile : string
                Path to the HDF5 file to load from.
        '''
        with h5py.File(inputFile, 'r') as hf:
            self.P                      = np.array(hf["populationMatrix"])
            self.currentGeneration, self.topRatioToRecombine, self.ratioToMutate = tuple(list(hf["metaInfo"]))
            self.psi                    = np.array(hf["networkArchitecture"])
            self.bestFitnessPerGen      = list(hf["bestFitnessPerGen"])

        self.currentGeneration = int(self.currentGeneration)
        self.m = self.P.shape[1]
        self.n = self.P.shape[0]
        self.l = int(self.m * self.topRatioToRecombine)
        self.markedForPermutation = self.initMarkedForPermutation()
        self.STOP = False

        # Below parameters are not essential to the working of CoSyNE
        self.lastImprovedGen = self.currentGeneration

        print("#### Resuming from generation {} with best fitness {} ####".format(self.currentGeneration, float(self.P[:,0,1].mean())))

    def evolve(self):
        ''' Evolves the population to the next generation. '''
        if self.STOP:
            return

        for j in range(self.m):
            X_fitness = self.evaluate(self.P[:, j, 0], self.psi)
            self.updateGenesFitness(j, X_fitness)

        # Doesn't mix weights between genotypes, instead keeps genotypes in one piece
        #self.sortGenotypesByMeanFitness(self.P, inplace=True)
        
        #'''
        self.bestFitnessPerGen.append(float(self.P[:,0,1].mean()))
        if self.currentGeneration > 2 and self.bestFitnessPerGen[-2] == self.bestFitnessPerGen[-1]:
            countLastImproved = self.currentGeneration - self.lastImprovedGen
            if self.verbose:
                print("==> {}/{} generations since last change".format(countLastImproved, self.lastImprovedGen), end="\r")
            if self.currentGeneration > 3 and countLastImproved > self.lastImprovedGen:
                if self.verbose:
                    print("More than {} generations since last improvement, stopping..".format(self.lastImprovedGen))
                self.STOP = True
        else:
            if self.verbose:
                print(" "*30, end="\r")
                print("Generation {}\t|\t".format(self.currentGeneration), end="")
                print("Top fitness changed : {}".format(self.bestFitnessPerGen[-1]))
            self.lastImprovedGen = self.currentGeneration
        #'''

        # Crossover then mutates
        O = self.recombine()

        for i in range(self.n):
            # Sort(P[i])
            # Mixes weights between genotypes
            self.sortSubpopulationGenesByFitness(i, inplace=True)

            # Calculate this once per population as it's expensive
            minFit = self.P[i, :, 1].min()
            maxFit = self.P[i, :, 1].max()

            # Replace the last l column by the children
            self.P[i, -self.l:] = O[i,:]

            for j in range(self.m):
                if random.random() < self.probability(fitness=self.P[i, j, 1], minFit=minFit, maxFit=minFit):
                    self.markedForPermutation[i,j] = 1
            self.permuteMarked(i)

        self.markedForPermutation = self.initMarkedForPermutation()
        self.currentGeneration += 1