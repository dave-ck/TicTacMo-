import time

from numba import jit
import numba
import numpy as np
from board import Board


@jit(nopython=True)
def playout(positions, num_games, lines, num_pos, turn, n, q):
    # make num_games copies of board in a 2D array
    boards = np.ones((num_games, num_pos)) * positions
    wins = np.zeros(num_games)  # 0 if active, Q if won by player Q
    while not wins.all() and turn < num_pos:
        player = (turn % q) + 1
        empties = (boards == 0).nonzero()
        for boardNo in range(num_games):
            if wins[boardNo] == 0:
                move = np.random.choice(empties[1][empties[0] == boardNo])
                boards[boardNo, move] = player  # make a random move
                # print("Player:", player, "made move:", move)
                # print("Resulting board:", boards[boardNo])
                # check for wins, and updated the wins array
                board_lines = boards[boardNo].take(lines)
                player_wins = ((board_lines == player).sum(axis=1) == n).any()
                if player_wins:
                    wins[boardNo] = player
        turn += 1
    return wins


b = Board.blank_board(3, 2, 3)

for n_g in [10, 100, 1000, 10000, 100000]:
    s = time.time()
    res = playout(b.positions, n_g, b.lines, b.num_pos, b.turn, b.n, b.q)
    e = time.time()
    print("Computed %d games in %5f seconds" % (n_g, e-s))
    print(np.unique(res, return_counts=True))
