from boost import Boost
import datareader
#from kfold import kfold
from random import shuffle

# Read dataset
games = datareader.readGames('./dataset/tic-tac-toe.data')

# Define K as in kfold
k = 5

# Define nstumps as in the number of weak classifiers that would be used
nstumps = 5

# Shuffle the dataset
shuffle(games)

# Begin Kfold by dividing dataset into k parts
subset_size = int(len(games)/k)

kfold_accuracy = 0.0

print("\033[1mEvaluating Adaboost with " +  str(k) + "-folds algorithm\033[0m")

for i in range(k):
    test_data = games[i*subset_size:][:subset_size]
    training_data = games[:i*subset_size] + games[(i+1)*subset_size:]

    print("\033[1mFold "+ str(i+1) + " \033[0m")
    # Define a Adaboost classifier
    boost = Boost(len(training_data),nstumps)
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


print("\033[95mThe final accuracy of boost with "+str(nstumps) + " weak classifiers is : " + str(kfold_accuracy))

