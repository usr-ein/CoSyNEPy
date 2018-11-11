import numpy as np
from cosyne import CoSyNE
import trainingLogger
import argparse

def main(maxGeneration, m, psi=[1,3,1], topRatioToRecombine=0.25, ratioToMutate=0.20, seed=None, verbose=False, outputFile=None):
    trainer = CoSyNE(m, psi, topRatioToRecombine, ratioToMutate, seed, verbose)
    for e in range(maxGeneration):
        trainer.evolve()
    if outputFile != None:
        trainingLogger.writeLog(outputFile=outputFile,
        lastImprovedGen     = trainer.lastImprovedGen,
        bestFitnessPerGen   = trainer.bestFitnessPerGen,
        n                   = trainer.n,
        m                   = trainer.m,
        psi                 = trainer.psi,
        topRatioToRecombine = trainer.topRatioToRecombine,
        ratioToMutate       = trainer.ratioToMutate,
        maxGeneration       = maxGeneration,
        seed                = seed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("maxGeneration", help="The number of generations to be evolved", type=int)
    parser.add_argument("m", help="The number of genotypes to be evolved", type=int)
    parser.add_argument('--psi', nargs='+', type=int, help="architecture of the fully connected netwok,\
                                                            as a list of neurone per layer. Example: --psi 1 3 1", default=[1,3,1])
    parser.add_argument("--recombine", help="The ratio of the best genotypes to be recombined into offsprings", type=float, default=0.25)
    parser.add_argument("--mutate", help="The ratio of the offsprings to be mutated", type=float, default=0.20)
    parser.add_argument("--seed", help="Seed to make experiments repeatable (int)", type=int, default=None)
    parser.add_argument("-o", "--output", help="Output path were to save the log file", default=None)
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")

    args = parser.parse_args()
    main(
            args.maxGeneration, 
            args.m, 
            args.psi, 
            args.recombine, 
            args.mutate, 
            args.seed, 
            args.verbose, 
            args.output
        )