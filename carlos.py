import numpy as np
from board import Board

b = Board.blank_board(3, 2, 2)

print(b.positions)
print(b.lines)
b.move(0)
b.move(1)
b.move(3)
b.move(5)
b.cli()
print(b.positions)
print(b.positions==0)
e = (b.positions==0).nonzero()[0]
boards5 = np.array([b.positions.copy() for _ in range(5)])
boards5[:, 8] = 1
boards5[3, 8] = 0
boards5[3, 6] = 1
print(boards5)
boards5_lines = boards5[:, b.lines]
print(boards5_lines)
player = 1
boards5_wins_player = ((boards5_lines == player).sum(axis=2) == b.n).any(axis=1)
print(boards5_wins_player)
