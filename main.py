from boost import Boost
import datareader


games = datareader.readGames('./dataset/tic-tac-toe.data')

boost = Boost(len(games),7)
boost_1 = Boost(len(games),1)
boost_2 = Boost(len(games),2)
boost_4 = Boost(len(games),4)
boost_5 = Boost(len(games),4)
boost_8 = Boost(len(games),8)
boost_16 = Boost(len(games),16)
boost_64 = Boost(len(games),64)

#print(boost.weights)
boost.build_classifier(games)
#boost_5.build_classifier(games[:10])
acertos = 0
erros =  0
#
for sample in games:
    #print(sample[9])
    #print("oi: " + str(boost.classify(sample)))
    if sample[9] == "positive" and boost.classify(sample) == 1:
        acertos += 1
        #print("acertei")
    elif sample[9] == "negative" and boost.classify(sample) == -1:
        acertos += 1
        #print("acertei")
    else:
        erros += 1
        #print("errei")
    #print("Classifiquei: " + str(boost.classify(sample)))

print("Certos: " + str(acertos) +", errors: " +str(erros) )
#result = 0.0
#for i in boost.weights:
#    result += i[0]
#
#print(boost.weights)
#print("Peso: " + str(result))

