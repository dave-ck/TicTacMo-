from game import Game
from random_play import Randy
from human import Human


k = 2
n = 6
q = 3

randy = Randy(k, n, q)
# human = Human(k, n, q)
game = Game(k, n, q, [randy, randy])    # randy gets first 2 moves, then computer gets a move

game.play()

