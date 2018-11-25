import numpy as np
import random

from cosyne import CoSyNE
import evaluators
from logger import Logger

def main():
    verbose = False
    graphing = False
    verboseResults = False

    seed = 0

    m = 1000
    psi = [1, 5, 3, 1]
    maxGen = 100
    maxFitness = None

    sampling = 3
    recombination = 0.20
    mutation = 0.05

    np.random.seed(seed)
    random.seed(seed)

    logger = Logger(m, graphing=graphing, verbose=verbose)
    evaluator = evaluators.Evaluator(targetFunc=lambda x : x*0.23, sampling=sampling, logger=logger)

    trainer = CoSyNE(m, psi, evaluator, topRatioToRecombine=recombination, ratioToMutate=mutation, maxFitness=maxFitness)

    for i in range(maxGen):
        if trainer.evolve():
            if verbose: print("#"*19 + " Max fitness reached " + "#"*19)
            break
    if verbose: print("#"*20 + " Evolving finished " + "#"*20)

    if verboseResults:
        pop = trainer.population

        logger.summary(pop.fitnesses, pop.currentGeneration)
        bestFitnessIndex = pop.fitnesses.mean().argmax()
        evaluator.run(pop.buildNetwork(bestFitnessIndex, psi), np.array([0.6923]))

if __name__ == '__main__':
    main()