#from termcolor import colored
import numpy as np
import matplotlib.pyplot as plt
import time
class Logger():
    """docstring for Logger"""

    def __init__(self, m, graphing=True, verbose=True):
        self.m = m
        self.graphing = graphing
        self.verbose = verbose
        # color specific to genotype
        self.colorMap = plt.cm.get_cmap('hsv', self.m)
        self.beginTime = time.time()

        self.lastText = None

    def summary(self, fitnesses, currentGen):
        # Verbose
        t = time.time()-self.beginTime
        print("Gen : {:4} | ^f : {:.3E} | ≈f : {:.3E} | ⏱️  {:.3E}s".format(currentGen, fitnesses.max(), fitnesses.mean(), t))

    def log(self, fitnesses, currentGen):
        # Graph
        if self.graphing:
            self.updateGraph((currentGen, fitnesses.mean()), 0)
        if self.verbose:
            self.summary(fitnesses, currentGen)

    def updateGraph(self, newPoint, j):
        plt.axis()
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # fig.subplots_adjust(top=0.85)
        plt.title('Avg fitness over generation of CoSyNE')
        plt.xlabel('Generation')
        plt.ylabel('Avg fitness')
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

        if self.lastText:
            self.lastText.remove()
        self.lastText = plt.text(0,0, 'Time Elapsed: {:.2f}s\n'.format(time.time()-self.beginTime), fontsize=10)

        plt.scatter(newPoint[0], newPoint[1], c=[self.colorMap(j)], s=4.2)
        plt.pause(0.005)
