import json

def writeLog(outputFile, lastImprovedGen, bestFitnessPerGen, n, m, psi, topRatioToRecombine, ratioToMutate, maxGeneration, seed=None):
    ''' Logs in JSON file format the provided informations about the run.

        Parameters
        ----------
        outputFile : string
            Path were to write the output logs, with filename.
        lastImprovedGen : int
            Number of the last improved generation.
        bestFitnessPerGen: list[float]
            List of each generation's best fitness were the index is the generation and the value the fitness.
        n: int
            The number of populations.
        m: int
            The number of genotypes.
        psi: list[int]
            The network architecture.
        topRatioToRecombine: float
            The ratio of the best genotypes to be recombined into offsprings>
        ratioToMutate: float
            The ratio of the offsprings to be mutated.
        seed: int, optional
            The seed with which was initialised the pseudorandom number generator (if any).
        maxGeneration: int
            The maximum number of generations to evolve.
    '''
    logDict = {
                'lastImprovedGen'       : lastImprovedGen, 
                'bestFitnessPerGen'     : bestFitnessPerGen, 
                'n'                     : n, 
                'm'                     : m, 
                'psi'                   : psi, 
                'topRatioToRecombine'   : topRatioToRecombine, 
                'ratioToMutate'         : ratioToMutate, 
                'maxGeneration'         : maxGeneration, 
                'seed'                  : seed
            }
    with open(outputFile, 'w') as outfile:
        json.dump(logDict, outfile)