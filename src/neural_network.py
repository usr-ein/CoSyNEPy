import numpy as np
# Fast sigmoid, slower for single value cause C overhead
from scipy.special import expit

class NeuralNetwork():
    '''Representation of a classical feed forward multi layer perceptron.

    This representation is shape agnostic, meaning that each layer can have a
    different shape as long as they're all fully connected (i.e. that the weight matrix is
    coherent).

    This class can run on multiple threads (tested)
    '''

    def __init__(self, weightMatrices, psi, activationFunctions=None, costFunction=None):
        '''Initialise the NeuralNetwork.
        Parameters
        ----------
        weightMatrices : np.ndarray
            Array of weight matrix. Each matrix represent the transition from Layer_i to Layer_i+1,
            hence the next matrix shall have the same number of rows as the number of columns of the previous.
            This coherence can and will be tested through the checkCoherenceWeights method.
        psi : np.ndarray
            Array of neurone count on each layer. For instance if the networks has 3 inputs, 1 hidden layer with 5 neurones
            and one hidden layer with 3 then 2 outputs, psi = np.array([3, 5, 3, 2]) . Use numpy array even if the list
            seems small, we won't modify psi so let's optimise it.
        activationFunctions : list[functions]
            List of the activation functions for each layer, default is relu until last layer, then sigmoid.
        costFunction : function
            Fuction to compute cost, should be like cost(pred, targets) --> float32. Default: RMS error
        '''
        # Array of matrices of size Layer_i x Layer_i+1
        # where lines contains weights from nth neurone in Layer_i to all the m neurones in Layer_i+1
        self.weightMatrices = weightMatrices
        self.psi = psi
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

        # Only when troubleshooting
        # self.checkCoherenceWeights()  # Verifies that all the weights are well defined


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
