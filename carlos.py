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

@jit(nopython=True)
def playout_ind(positions, num_games, lines, num_pos, turn, n, q):
    # make num_games copies of board in a 2D array
    positions.astype(np.int16)
    wins = np.zeros(num_games, dtype=np.uint16)  # 0 if active, Q if won by player Q
    base_turn = turn
    for boardNo in range(num_games):
        board = positions.copy()
        turn = base_turn
        while not wins[boardNo] and turn < num_pos:
            player = (turn % q) + 1
            empties = (board == 0).nonzero()[0]
            move = np.random.choice(empties)
            board[move] = player
            board_lines = board.take(lines)
            player_wins = ((board_lines == player).sum(axis=1) == n).any()
            if player_wins:
                wins[boardNo] = player
            turn += 1
    return wins

@jit(nopython=True)
def playout_hyb(positions, num_games, lines, num_pos, turn, n, q):
    num_batches = int(num_games / 128)
    out = np.empty((num_batches, 128), dtype=np.uint16)
    for index in range(num_batches):
        out[index] = playout_ind(positions, 128, lines, num_pos, turn, n, q)
    return out




# b = Board.blank_board(4, 3, 4)
# mgps = 0
# for n_g in [4096, 8192, 16384, 32768, 65536]:
#     print("%d games:" % n_g)
#     s = time.time()
#     res = playout(b.positions, n_g, b.lines, b.num_pos, b.turn, b.n, b.q)
#     e = time.time()
#     secs = e-s
#     print("C: Done in %5f seconds" % secs)
#     gps = n_g / secs
#     mgps = max(mgps, gps)
#     print("C: Games per second: %5f" % (n_g/secs))
#     print("C:", dict(zip(*np.unique(res, return_counts=True))))
#     try:
#         s = time.time()
#         res = playout(b.positions, n_g, b.lines, b.num_pos, b.turn, b.n, b.q)
#         e = time.time()
#         secs = e-s
#         print("C_i: Done in %5f seconds" % secs)
#         gps = n_g / secs
#         mgps = max(mgps, gps)
#         print("C_i: Games per second: %5f" % (n_g/secs))
#         print("C_i:", dict(zip(*np.unique(res, return_counts=True))))
#     except ZeroDivisionError as e:
#         print("C_i too damn good, zerodiverror", e)
#     s = time.time()
#     res = b.playout_plain(n_g)
#     e = time.time()
#     secs = e - s
#     print("B: Done in %5f seconds" % secs)
#     gps = n_g / secs
#     mgps = max(mgps, gps)
#     print("B: Games per second: %5f" % gps)
#     print("B:", res)
#     try:
#         s = time.time()
#         res = playout_hyb(b.positions, n_g, b.lines, b.num_pos, b.turn, b.n, b.q)
#         e = time.time()
#         secs = e - s
#         print("C_h: Done in %5f seconds" % secs)
#         gps = n_g / secs
#         mgps = max(mgps, gps)
#         print("C_h: Games per second: %5f" % gps)
#         print("C_h:", dict(zip(*np.unique(res, return_counts=True))))
#     except ZeroDivisionError as e:
#         print("C_h too damn good, zerodiverror", e)
#
# print("mgps:", mgps)