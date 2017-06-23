# Run KFold using a Boost class on dataset
def kfold(classifier,k,dataset):
    rights = 0
    wrongs =  0
    for sample in dataset:
        if sample[9] == "positive" and classifier.classify(sample) == 1:
            rights+= 1
        elif sample[9] == "negative" and classifier.classify(sample) == -1:
            rights+= 1
        else:
            wrongs+= 1
    return [rights,wrongs]

