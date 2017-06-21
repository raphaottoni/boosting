import csv

# Read Tic Tac Toe data
def readGames(file_path):
   games = []
   with open(file_path) as csvfile:
       games_in_file = csv.reader(csvfile)
       for game in games_in_file:
           games.append(game)
       return games
