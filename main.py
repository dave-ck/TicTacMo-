from board import Board

# as used to prove the property of 3^3
b = Board.blank_board(3, 3, 4)
weights = {0: -10, 1: -10, 2: 2, 3: 1, 4: 1}
b.rev_guided_tree('carlo', 1, weights, bound=1024)


# to use a greedy guide instead, and try on 3^3 with 5 players
b = Board.blank_board(3, 3, 5)
weights = {0: -10, 1: -10, 2: 5, 3: 1, 4: 1, 5: 1}
b.rev_guided_tree('greedy', 1, weights, bound=1024)


# RL guide
b = Board.blank_board(3, 3, 2) # RL agent not trained for 3 players
weights = {0: -10, 1: -10, 2: 5} # unused, but must be passed as parameter
b.rev_guided_tree('rl', 1, weights, bound=1024)
