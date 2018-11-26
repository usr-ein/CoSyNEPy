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

    def evaluate(self, network):
        costs = 0
        for i in range(self.sampling):
            inputs = np.random.rand(network.psi[0])
            targetVal = self.targetFunc(inputs)
            predictions = network.forward(inputs)
            costs += network.costFunction(predictions, targetVal)

        return self.sampling/costs