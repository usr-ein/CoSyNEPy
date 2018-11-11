import numpy as np
from cosyne import CoSyNE

def main():
    trainer = CoSyNE(800, [1,3,1], topRatioToRecombine=0.25, ratioToMutate=0.20, verbose=True)
    for e in range(1000):
        trainer.evolve()

if __name__ == '__main__':
    main()