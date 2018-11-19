import numpy as np
import argparse
import h5py
from neural_network import NeuralNetwork
import evaluators

def main(inputs, inputFile):
    weightMatrices, psi = [], None
    with h5py.File(inputFile, 'r') as hf:
        psi = np.array(hf["networkArchitecture"])
        for i in range(len(psi)-1):
            weightMatrices.append(np.array(hf["weightMatrices/{}".format(i)]))
    nn = NeuralNetwork(np.array(weightMatrices), psi)

    #evaluator = evaluators.FooEval(0.5)

    evaluator = evaluators.Rosenbrock()
    evaluator.run(nn, np.array(inputs))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("load", help="Load a saved state of the network")
    parser.add_argument('inputs', nargs='+', type=float, help="Inputs for the network")

    args = parser.parse_args()
    main(args.inputs, args.load)