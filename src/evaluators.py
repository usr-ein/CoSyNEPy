import numpy as np
import helpers

class Evaluator():
    def __init__(self, targetFunc, sampling=10, logger=None):
        self.sampling = sampling
        self.costFunction = helpers.rmse
        self.targetFunc = np.vectorize(targetFunc)
        self.logger = logger

    def runTest(self, network, inputs):
        assert inputs.shape[0] == network.psi[0]
        targetVal = self.targetFunc(inputs)
        predictions = network.forward(inputs)
        cost = network.costFunction(predictions, targetVal)
        print("Inputs : \t", inputs)
        print("Target : \t", targetVal)
        print("Preds  : \t", predictions)
        print("Cost   : \t", cost)
        print("Fitness: \t", 1/cost)

    def evaluatePop(self, population, psi):
        nets = [population.buildNetwork(j, psi) for j in range(population.m)]
        fitnesses = np.array([self.evaluate(net) for net in nets])

        if self.logger: self.logger.log(fitnesses, population.currentGeneration)

        return fitnesses

    def evaluate(self, net):
        # a bit faster than for loop
        vfunc = np.vectorize(net.forward)

        inputs = np.random.rand(net.psi[0], self.sampling)
        targetVals = self.targetFunc(inputs)
        predictions = vfunc(inputs)
        costs = net.costFunction(predictions, targetVals)

        return 1/costs

