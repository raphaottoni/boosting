from boost import Boost
import datareader


games = datareader.readGames('./dataset/tic-tac-toe.data')

boost_2 = Boost(len(games),2)
boost_4 = Boost(len(games),4)
boost_8 = Boost(len(games),8)
boost_16 = Boost(len(games),16)

#boost_2.execute(games)
boost_16.execute(games)
