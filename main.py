from game import Game
from random_play import Randy
from phils_dq import TF_Player

k = 2
n = 3
q = 2

randy = Randy(k, n, q)
tf = TF_Player(path="models/p0_q2_n3_k2_games200_fc11024_fc21024_alpha0.003_.pth")
game = Game(k, n, q, [randy, None])  # randy gets first 2 moves, then computer gets a move

game.play()
