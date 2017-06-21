import numpy as np
import copy
import math

class Boost:

    # Class constructor
    def __init__(self,nsamples,nstumps):
        print("Creating boosting algorithm with " + str(nsamples)+ " samples and " + str(nstumps) + " stubs" )
        self.weights = np.full((nsamples,1),1/nsamples)
        self.nstumps = nstumps
        self.chosen_stumps = []
        self.chosen_alphas = []
        self.chosen_errors = []
        # Build stumps available on Tic Tac Toe
        self.stumps = self.buildStumps()


    #Calculate the alpha regarding the error
    def calculate_alpha(self,error):
        return 1/2 * math.log((1-error)/(error))

    # Run the boosting algorithm for the class
    def execute(self,samples):

        # runs n times, where n is the number of desirable stumps
        for i in range(self.nstumps):
            self.choose_stump(samples)
            self.update_weights(self.chosen_stumps[-1],samples)

        print(self.chosen_stumps)
        print(self.chosen_errors)
        print(self.chosen_alphas)


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

    # Finds the next best stump candidate
    # Returns the stump and the error associated
    def choose_stump(self,samples):
        stump_choosen_id = -1
        min_error = None

        # iterate over all possible stumps and chose the local best
        for characteristic_id, (characteristic,value) in enumerate(self.stumps):

            error = 0.0
            # iterate over all samples
            for i,sample in enumerate(samples):
                if (sample[characteristic] == value and sample[9] == "negative" ):
                  error += self.weights[i][0]
                elif (sample[characteristic] != value and sample[9] == "positive" ):
                  error += self.weights[i][0]

            # Keep the lowest error found
            if min_error is None:
                min_error = error
                stump_choosen_id = characteristic_id
            elif min_error > error:
                min_error = error
                stump_choosen_id = characteristic_id

        print("[Chosen] characteristic " + str(stump_choosen_id) + " errou : " + str(min_error) )

        # Add the selected stumps statistics to the chosen arrays
        self.chosen_stumps.append(stump_choosen_id)
        self.chosen_errors.append(min_error)
        self.chosen_alphas.append(self.calculate_alpha(min_error))



    # Update weight function regarding the chosen stump
    def update_weights(self,stump_id,samples):

        #old_weights =  copy.copy(self.weights)

        # Get the stump attributes
        characteristic,value = self.stumps[stump_id]
        error = self.chosen_errors[-1]


        # Iterate over each sample to adjust its weight
        for i,sample in enumerate(samples):
          if (sample[characteristic] == value and sample[9] == "negative" ):
            self.weights[i][0] = self.weights[i][0]/2 * 1/error
          elif (sample[characteristic] != value and sample[9] == "positive" ):
            self.weights[i][0] = self.weights[i][0]/2 * 1/error
          else:
            self.weights[i][0] = self.weights[i][0]/2 * 1/(1 - error)
          #print("new weight: " + str(self.weights[i][0]))





