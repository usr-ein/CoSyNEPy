import numpy as np
#from scipy.optimize import rosen as rosenSciPy # The Rosenbrock function
import gym

class FooEval():
    """Simplest evaluator possible. Evolves the network to multiply by self.factor.
    Factor should be less or equal to one and positive for the network's weights and outputs are scaled between 0 and 1."""
    def __init__(self, factor):
        assert factor >= 0
        assert factor <= 1
        self.factor = factor
        self.samplings = 10

    def run(self, network, inputs):
        assert inputs.shape[0] == network.psi[0]
        target = inputs * self.factor
        predictions = network.forward(inputs)
        cost = network.costFunction(predictions, target)
        print("Inputs : \t", inputs)
        print("Target : \t", target)
        print("Preds  : \t", predictions)
        print("Cost   : \t", cost)
        print("Fitness: \t", 1/cost)

    def evaluator(self, network):
        ''' Evaluate the given network on its ability to perform a multiplication by self.factor.
        The network should have the same amout of inputs as outputs.
        The network will evolve to approximate the multiplication of each inputs by self.factor.

        Parameters
        ----------
        network : NeuralNetwork
            The neural network to be evaluated, with pre-loaded weights and preloaded costFunction.
        '''
        fitness = 0
        for i in range(self.samplings):
            inputs = np.random.rand(network.psi[0])
            target = inputs * self.factor

            predictions = network.forward(inputs)
            cost = network.costFunction(predictions, target)
            fitness += 1/cost # we try to increase the fitness, so we try to increase the inverse of the cost
            
        return fitness/self.samplings

class Rosenbrock():
    """Evaluate the given network onto the Rosenbrock function (two variables).
    The network should have the an even amount of inputs and half that amount of output, e.g. 4->2 or 6->3 or 24->12.
    The network will evolve to approximate the function at n random points.
    The first half of the inputs are the x variable, the second half is the y variable."""
    def __init__(self):
        self.rosenBivariate = lambda x,y: (1-x) ** 2 + 100*(y-x**2)**2
        self.firstTime = True # to check if the network is well setup the first time around
        self.inputSize = 0 # will contain the size of the inputX, inputY to be generated, = psi[0]/2
        self.samplings = 10 #number of times the network will be evaluated

    def debug(self):
        preds = np.array(self.preds)
        print(preds[:,0].max(), preds[:,1].max(), "\n####")
        print(preds[:,0].min(), preds[:,1].min(), "\n####")

    def run(self, network, inputs):
        assert inputs.shape[0]%2 == 0
        half = int(inputs.shape[0]/2)
        inputs_X = inputs[:half]
        inputs_Y = inputs[half:]
        assert inputs_X.shape == inputs_Y.shape
        print("Rosenbrock(x = {}, y = {})".format(inputs_X, inputs_Y))
        target = self.rosenBivariate(inputs_X, inputs_Y)
        inputs_XY = np.concatenate( (inputs_X, inputs_Y) )
        predictions = network.forward(inputs_XY)
        predictions *= 101
        cost = network.costFunction(predictions, target)
        print("Target : \t", target)
        print("Preds  : \t", predictions)
        print("Cost   : \t", cost)
        print("Fitness: \t", 1/cost)
        return

    def evaluator(self, network):
        ''' Evaluate the given network onto the Rosenbrock function.
        The network should have the an even amount of inputs and half that amount of output, e.g. 4->2 or 6->3 or 24->12.
        The network will evolve to approximate the function at n random points.
        The first half of the inputs are the x variable, the second half is the y variable.

        Parameters
        ----------
        network : NeuralNetwork
            The neural network to be evaluated, with pre-loaded weights and preloaded costFunction.
        '''

        if self.firstTime:
            assert network.psi[0] % 2 == 0 # First layer should have an even number of neuron, see this function's doctstring
            assert network.psi[0]/2 == network.psi[-1] # Last layer should be half of the first layer, see this function's doctstring
            self.firstTime = False
            self.inputSize = int(network.psi[0]/2)

        fitness = 0
        for i in range(self.samplings):
            inputs_X = np.random.rand(self.inputSize)
            inputs_Y = np.random.rand(self.inputSize)

            target = self.rosenBivariate(inputs_X, inputs_Y)

            inputs_XY = np.concatenate( (inputs_X, inputs_Y) )
            # for x, y in [0, 1]
            # max(rosenBivariate(x,y)) == 101 when {x: 0, y: 1}
            # min(rosenBivariate(x,y)) == 0 when {x: 1, y: 1}
            # hence
            predictions *= 101
            cost = network.costFunction(predictions, target)
            # we try to increase the fitness, so we try to increase the inverse of the cost
            fitness += (1/cost)
        return fitness/self.samplings

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