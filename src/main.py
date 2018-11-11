import numpy as np
from cosyne import CoSyNE

def main():
    np.seterr(all='raise')
    trainer = CoSyNE(2000, [1,3,1], topRatioToRecombine=0.25, ratioToMutate=0.20, verbose=1)
    for e in range(3000):
        trainer.evolve()

if __name__ == '__main__':
    main()