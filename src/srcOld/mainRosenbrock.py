import numpy as np
from cosyne import CoSyNE
import trainingLogger
import argparse
import evaluators
from helpers import askIfExportState, askIfExportNetwork

def main(maxGeneration, m, psi=[1,3,1], topRatioToRecombine=0.25, ratioToMutate=0.20, seed=None, verbose=False, loadFile=None, logFile=None):
    rosenEval = evaluators.Rosenbrock()
    trainer = CoSyNE(m=m, psi=psi, evaluator=rosenEval.evaluator, topRatioToRecombine=topRatioToRecombine, ratioToMutate=ratioToMutate, seed=seed, verbose=verbose, loadFile=loadFile)

    if loadFile != None:
        trainer.importCurrentGeneration(loadFile)
    try:
        for e in range(maxGeneration):
            trainer.evolve()
    except KeyboardInterrupt as e:
        askIfExportState(trainer)
        askIfExportNetwork(trainer)
    rosenEval.debug()
    askIfExportNetwork(trainer)
    if logFile != None:
        trainingLogger.writeLog(outputFile=logFile,
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
    parser.add_argument("-m", help="The number of genotypes to be evolved", type=int)
    parser.add_argument('--psi', nargs='+', type=int, help="architecture of the fully connected netwok,\
                                                            as a list of neurone per layer. Example: --psi 1 3 1", default=[1,3,1])
    parser.add_argument("--recombine", help="The ratio of the best genotypes to be recombined into offsprings", type=float, default=0.25)
    parser.add_argument("--mutate", help="The ratio of the offsprings to be mutated", type=float, default=0.20)
    parser.add_argument("-s", "--seed", help="Seed to make experiments repeatable (int)", type=int, default=None)
    parser.add_argument("-o", "--outputLog", help="Output path were to save the log file", default=None)
    parser.add_argument("-l", "--load", help="Load a saved state of the network", default=None)
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")

    args = parser.parse_args()
    main(maxGeneration=args.maxGeneration,
        m=args.m,
        psi=args.psi,
        topRatioToRecombine=args.recombine,
        ratioToMutate=args.mutate,
        seed=args.seed,
        verbose=args.verbose,
        loadFile=args.load,
        logFile=args.outputLog
    )
