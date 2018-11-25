from termcolor import colored
import numpy as np
import matplotlib.pyplot as plt
class Logger():
    """docstring for Logger"""
    def __init__(self, m):
        self.m = m
        #color specific to genotype
        self.colorMap = plt.cm.get_cmap('hsv', self.m)

    def log(self, fitnesses, currentGen):
        # Verbose
        #print("Generation {}\t|\tTop fitness : {}".format(currentGen, fitnesses.max().round(decimals=3)), end="")
        # Graph
        if currentGen%3 == 0:
            self.updateGraph((currentGen, fitnesses.mean()), 0)

    def updateGraph(self, newPoint, j):
        plt.axis()
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # fig.subplots_adjust(top=0.85)
        plt.title('Avg fitness over generation of CoSyNE')
        plt.xlabel('Generation')
        plt.ylabel('Avg fitness')

        # ax.text(2, 6, 'blah', fontsize=10)


        plt.scatter(newPoint[0], newPoint[1], c=[self.colorMap(j)], s=4.2)
        plt.pause(0.005)