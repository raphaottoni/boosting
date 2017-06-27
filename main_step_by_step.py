from boost import Boost
import datareader
#from kfold import kfold
from random import shuffle
import argparse

# Open file to create csv
output = open("results.csv","w" )
output.write("nstumps,test_error,training_error,stump_error,stump_alpha\n")



# Create parser
parser = argparse.ArgumentParser(description='Adaboost algorithm for Tic Tac Toe dataset')
parser.add_argument('nstumps', metavar='nstumps', type=int, default=5,
                            help='The number of Weak Classifiers that will be used')

#Parse
args = parser.parse_args()


# Read dataset
games = datareader.readGames('./dataset/tic-tac-toe.data')

# Define K as in kfold
k = 5

# Define nstumps as in the number of weak classifiers that would be used
nstumps = args.nstumps

# Shuffle the dataset
shuffle(games)

# Begin Kfold by dividing dataset into k parts
subset_size = int(len(games)/k)

# Warning
print("\033[93mThis algorithm should be executed with python3, otherwise it WONT work!\033[0m")

# Run k-fold for each one of the nWeakClassifers
for n in range(nstumps):

    kfold_accuracy = 0.0
    kfold_accuracy_traning = 0.0
    stump_error = 0.0
    stump_alpha = 0.0

    print("\033[1mEvaluating Adaboost with " +  str(k) + "-folds algorithm\033[0m")
    for i in range(k):

        test_data = games[i*subset_size:][:subset_size]
        training_data = games[:i*subset_size] + games[(i+1)*subset_size:]

        print("\033[1mFold "+ str(i+1) + " \033[0m")
        # Define a Adaboost classifier
        boost = Boost(len(training_data),n+1)
        # Train Adaboost classifier with the training data
        boost.build_classifier(training_data)

        # Validate classifier with test_data
        rights = 0
        wrongs =  0
        for sample in test_data:
            if sample[9] == "positive" and boost.classify(sample) == 1:
                rights+= 1
            elif sample[9] == "negative" and boost.classify(sample) == -1:
                rights+= 1
            else:
                wrongs+= 1
        kfold_accuracy += (rights/(rights+wrongs)*1/k)
        stump_error += boost.chosen_errors[-1] * (1/5)
        stump_alpha += boost.chosen_alphas[-1] * (1/5)

        # Validate classifier with training data
        rights = 0
        wrongs =  0
        for sample in training_data:
            if sample[9] == "positive" and boost.classify(sample) == 1:
                rights+= 1
            elif sample[9] == "negative" and boost.classify(sample) == -1:
                rights+= 1
            else:
                wrongs+= 1
        kfold_accuracy_traning += (rights/(rights+wrongs)*1/k)


    print("\033[95mThe final accuracy of boost with "+str(n+1) + " weak classifiers is : " + str(kfold_accuracy))
    output.write(str(n +1)+ ","+ str(1.0 - kfold_accuracy) +"," + str(1.0 - kfold_accuracy_traning) + ","+ str(stump_error) + "," + str(stump_alpha) + "\n")

output.close

