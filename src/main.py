import numpy as np

from cosyne import CoSyNE
import evaluators
from logger import Logger

def main():
    m = 1000
    psi = [1, 3, 5, 4, 1]
    maxGen = 300
    maxFitness = 10**20

    sampling = 5
    recombination = 0.25
    mutation = 0.10

    logger = Logger(m)
    evaluator = evaluators.Evaluator(targetFunc=lambda x : x*0.2, sampling=sampling, logger=logger)

    trainer = CoSyNE(m, psi, maxFitness, evaluator, topRatioToRecombine=recombination, ratioToMutate=mutation)

    for i in range(maxGen):
        if trainer.evolve():
            break

    pop = trainer.population
    bestFitnessIndex = pop.fitnesses.mean().argmax()
    evaluator.run(pop.buildNetwork(bestFitnessIndex, psi), np.array([0.6923]))

if __name__ == '__main__':
    main()