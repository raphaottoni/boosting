import numpy as np
import copy
import csv

class Boost:

    # Class constructor
    def __init__(self,nsamples,nstubs,dataset):
        print("Creating boosting algorithm with " + str(nsamples)+ " samples and " + str(nstubs) + " stubs" )
        self.weights = np.full((1,nsamples),1/nsamples)
        self.old_weights =  copy.copy(self.weights)
        self.alpha = 0.0
        self.error = 0.0
        self.chosen_stumps = []
        self.chosen_alphas = []
        self.chosen_errors = []

        # Build stumps available on Tic Tac Toe
        self.stumps = self.buildStumps()

    # Generator for Tic Tac Toe possible stumps
    def buildStumps(self):

        stumps = []
        # adding two inicial stumps for all positive and all negative
        #stumps.append([0,'true'])

        # for each position in Tic Tac Toe matrix
        for i in range(9):
            #For each label possibility
            for j in ['o','x','b']:
                stumps.append([i,j])
        return stumps



