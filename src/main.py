import numpy as np
from cosyne import CoSyNE

def main():
    trainer = CoSyNE(20, [1,3,1], topRatioToRecombine=0.25, ratioToMutate=0.20, verbose=4)
    for e in range(3):
        trainer.evolve()

if __name__ == '__main__':
    main()