import numpy as np
from scipy.optimize import rosen as rosenSciPy # The Rosenbrock function

def rosenbrock(network):
    ''' Evaluate the given network onto the Rosenbrock function.
    The network should have the same amout of output as inputs.
    The network will evolve to approximate the function at random points.

    Parameters
    ----------
    network : NeuralNetwork
        The neural network to be evaluated, with pre-loaded weights and preloaded costFunction.
    '''

    # assert psi[0] == psi[-1] # For the rosenbrock
    inputs = np.random.rand(network.psi[0])
    targets = rosenSciPy(inputs)

    predictions = network.forward(inputs)
    cost = network.costFunction(predictions, targets)

    return cost