import numpy as np
import time
from grapher import Grapher
class Logger():
    """docstring for Logger"""

    def __init__(self, m, graphing=True, verbose=True):
        self.m = m
        self.graphing = graphing
        self.verbose = verbose
        self.beginTime = time.time()

        if self.graphing:
            self.grapher = Grapher(title='Avg fitness over generation of CoSyNE', xlabel='Generation', ylabel='Avg fitness')
        self.lastText = None

    def summary(self, fitnesses, currentGen):
        # Verbose
        t = time.time()-self.beginTime
        print("Gen : {:4} | ^f : {:.3E} | ≈f : {:.3E} | ⏱️  {:.3E}s".format(currentGen, fitnesses.max(), fitnesses.mean(), t))

    def log(self, fitnesses, currentGen):
        # Graph
        if self.graphing:
            self.grapher.updateLabel('Time Elapsed: {:.2f}s\n'.format(time.time()-self.beginTime))
            self.grapher.newPoint(currentGen, fitnesses.mean())
        if self.verbose:
            self.summary(fitnesses, currentGen)
