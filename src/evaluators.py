import numpy as np
from scipy.optimize import rosen as rosenSciPy # The Rosenbrock function
import gym

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

class PoleBalancing():
    """Initialises the CartPole-v0 Gym environment"""
    def __init__(self, timesteps=100, render=False, verbose=False):
        self.env = gym.make('CartPole-v0')
        self.timesteps = timesteps
        self.render = render
        self.verbose = verbose

    def evaluator(self, network):
        observation = self.env.reset()
        avgReward = 0
        for t in range(self.timesteps):
            if self.render:
                self.env.render()
            action = 1 if network.forward(observation) > 0.5 else 0
            observation, reward, done, info = self.env.step(action)
            avgReward += reward
            if done:
                if self.verbose:
                    print("Episode finished after {} timesteps".format(t+1))
                break
        return avgReward/self.timesteps