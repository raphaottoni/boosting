import numpy as np
import math

class Boost:

    # Class constructor
    def __init__(self,nsamples,nstumps):
        print("Creating boosting class for  " + str(nsamples)+ " samples and " + str(nstumps) + " stubs" )
        self.weights = np.full((nsamples,1),1.0/nsamples)
        self.nstumps = nstumps
        self.nsamples = nsamples
        self.chosen_stumps = []
        self.chosen_alphas = []
        self.chosen_errors = []
        # Build stumps available on Tic Tac Toe
        self.stumps = self.buildStumps()


    #Calculate the alpha regarding the error
    def calculate_alpha(self,error):
        return 1/2 * math.log((1-error)/(error))

    # Run the boosting algorithm for the class
    def build_classifier(self,samples):

        # runs n times, where n is the number of desirable stumps
        for i in range(self.nstumps):
            self.choose_stump(samples)
            self.update_weights(self.chosen_stumps[-1],samples)

        print("Chosen Stumps order: "+ str(self.chosen_stumps))
        print("Stumps Errors: "+ str(self.chosen_errors))
        print("Stumps Alphas: " +str(self.chosen_alphas))


    # Classify an sample
    def classify(self,sample):
        class_sample = 0.0

        #iterate over the number of stumps chosen
        for i in range(len(self.chosen_stumps)):

            predict_stump_class = 0.0
            # get stump attributes
            characteristic,value,prediction = self.stumps[self.chosen_stumps[i]]

            # find the stump classification for this sample
            if (sample[characteristic] == value ):
              predict_stump_class = 1 if prediction == "positive" else -1
            else:
              predict_stump_class = -1 if prediction == "positive" else 1

            class_sample += self.chosen_alphas[i]*predict_stump_class

        resultado =  1 if class_sample > 0.0 else -1
        return resultado

    # Generator for Tic Tac Toe possible stumps
    def buildStumps(self):

        stumps = []

        # for each position in Tic Tac Toe matrix
        for i in range(9):

            # Create  stumps both for positive predictions and negative predictions
            for k in ['positive','negative']:
                #For each label possibility
                for j in ['o','x','b']:
                    stumps.append([i,j,k])
        return stumps

    # Finds the next best stump candidate
    def choose_stump(self,samples):
        stump_choosen_id = -1
        min_error = None

        # iterate over all possible stumps and chose the local best
        for characteristic_id, (characteristic,value, prediction) in enumerate(self.stumps):

            #if characteristic_id not in self.chosen_stumps:
            error = 0.0

            # iterate over all samples
            for i,sample in enumerate(samples):
                #print(sample[characteristic],value,sample[9])
                if (sample[characteristic] == value and sample[9] != prediction ):
                  error += self.weights[i][0]
                elif (sample[characteristic] != value and sample[9] == prediction ):
                  error += self.weights[i][0]

            # Keep the lowest error found
            if min_error is None:
                min_error = error
                stump_choosen_id = characteristic_id
            elif min_error > error:
                #print("troquei")
                min_error = error
                stump_choosen_id = characteristic_id

        #print("[Chosen] characteristic(" + str(stump_choosen_id) + ")["+str(self.stumps[stump_choosen_id][0]) + ","  + str(self.stumps[stump_choosen_id][1]) + "," + str(self.stumps[stump_choosen_id][2])  +"]   error: " + str(min_error) )


        # Add the selected stumps statistics to the chosen arrays
        self.chosen_stumps.append(stump_choosen_id)
        self.chosen_errors.append(min_error)
        self.chosen_alphas.append(self.calculate_alpha(min_error))



    # Update weight function regarding the chosen stump
    def update_weights(self,stump_id,samples):

        # Get the stump attributes
        characteristic,value,prediction = self.stumps[stump_id]
        error = self.chosen_errors[-1]


        # Iterate over each sample to adjust its weight
        for i,sample in enumerate(samples):
          if (sample[characteristic] == value and sample[9] != prediction ):
            self.weights[i][0] = (self.weights[i][0]/2) * (1/error)
          elif (sample[characteristic] != value and sample[9] == prediction ):
            self.weights[i][0] = (self.weights[i][0]/2) * (1/error)
          else:
            self.weights[i][0] = (self.weights[i][0]/2) * (1/(1 - error))





