import os
from board import Board
import numpy as np

b = Board.blank_board(3, 3, 3)
weights = {0: -1, 1: -1, 2: 1, 3: 1}
weights_arr = np.array(list(weights.values()), dtype=np.int8)
winner = b.win()
while not winner:
    if (b.turn % b.q) == 0:  # on human turn
        os.system('cls')
        b.human_move()
        print("Processing computer move...")
    else:
        b.carlo_greedy(1024, weights)
    winner = b.win()
os.system('cls')
print("Game won by player %d! Final configuration:" % winner)
b.cli()
