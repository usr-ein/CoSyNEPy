import numpy as np
import random

from cosyne import CoSyNE
import evaluators
from logger import Logger

def main():
    verbose = True
    graphing = True
    verboseResults = True

    seed = 0

    m = 1500
    psi = [1, 1]
    maxGen = 10000
    maxFitness = None

    sampling = 3
    recombination = 0.25
    mutation = 0.10

    np.random.seed(seed)
    random.seed(seed)

    logger = Logger(m, graphing=graphing, verbose=verbose)
    evaluator = evaluators.Evaluator(targetFunc=lambda x : (x*0.5), sampling=sampling, logger=logger)

    trainer = CoSyNE(m, psi, evaluator, topRatioToRecombine=recombination, ratioToMutate=mutation, maxFitness=maxFitness)
    try:
        for i in range(maxGen):
            if trainer.evolve():
                if verbose: print("#"*19 + " Max fitness reached " + "#"*19)
                break
            if i % 10 == 0:
                pop = trainer.population
                bestFitnessIndex = pop.fitnesses.mean().argmax()
                bestXweight = pop.P[:, bestFitnessIndex]
                bestXbiases = pop.biases[:, bestFitnessIndex]
                print("Weight = {}, bias = {}".format(bestXweight[0], bestXbiases[0]))
        if verbose: print("#"*20 + " Evolving finished " + "#"*20)
    except KeyboardInterrupt as e:
        pass

    if verboseResults:
        pop = trainer.population
        bestFitnessIndex = pop.fitnesses.mean().argmax()
        net = pop.buildNetwork(bestFitnessIndex, psi)

        logger.summary(pop.fitnesses, pop.currentGeneration)
        evaluator.runTest(net, np.array([0]))
        evaluator.runTest(net, np.array([0.2]))
        evaluator.runTest(net, np.array([0.6]))

        logger.grapher.plotFuncs([np.vectorize(net.forward), evaluator.targetFunc])

if __name__ == '__main__':
    main()