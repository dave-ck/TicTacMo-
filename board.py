import os

import numba
from numba import jit
import time
import random
import numpy as np
import torch
from dq import DeepQNetwork
from copy import deepcopy, copy


class Board:  # Cimpl entire class as a struct, functions as methods taking the struct as a parameter

    def __init__(self, n, k, q, num_pos, positions, lines, mappings, turn, RL_models=None):
        self.name = "Game"
        self.n = n  # number of
        self.k = k  # number of dimensions
        self.q = q  # number of players
        self.num_pos = num_pos  # precomputed and passed as param for speed, p = n**k
        self.turn = turn  # specific to configuration
        self.lines = lines
        self.num_lines = len(self.lines)
        self.mappings = mappings
        self.positions = positions  # 0 for empty, 1 for X (first and odd moves), -1 for O (second and even moves)
        self.RL_models = RL_models or {i: None for i in range(1, q + 1)}

    @classmethod
    def blank_board(cls, n, k, q):
        num_pos = n ** k
        positions = np.zeros(num_pos, dtype=np.int64)
        lines = generate_lines(n, k)
        mappings = generate_transforms(n, k)
        return cls(n, k, q, num_pos, positions, lines, mappings, turn=0)

    def reset(self):
        self.turn = 0
        self.positions = np.zeros(self.num_pos, dtype=np.int64)

    def successors(self):  # Cimpl with isomorphism checks
        """
        Produces all possible successors of the Board configuration (which are 1 move away). Does not reduce these.
        :return: A set containing Board objects.
        """
        successors = set()
        for index in range(self.num_pos):
            if self.positions[index] == 0:
                successors.update({self.move_clone(index).reduce()})
        for successor in successors.copy():
            for equiv in successors.copy():
                if equiv.equals(successor):
                    successors.remove(equiv)
            successors.add(successor)  # otherwise just empties entire collection
        return successors

    def win(self):
        """

        :return: number of winning player if some player wins; -1 if a draw; 0 otherwise
        """
        return win_(self.n, self.k, self.q, self.positions, self.lines, self.num_pos, self.num_lines)

    def move(self, position):  # return a copy of the config, with the move added
        if self.positions[position] == 0:  # illegal move are ignored, turn incremented and player loses their move
            self.positions[position] = self.turn % self.q + 1
        else:
            raise ValueError("Invalid move issued")
        self.turn += 1

    def move_clone(self, position):
        """
        Clones the current Board, makes the move on the clone, then returns the clone.
        :param position: The position, as an integer, where the move should be made (i.e. 0 for top left corner, in 2D)
        :return: the cloned Board with the move applied.
        """
        clone_board = Board(self.n, self.k, self.q, self.num_pos, self.positions.copy(), self.lines, self.mappings,
                            self.turn, self.RL_models)
        clone_board.move(position)
        return clone_board

    def reduce(self):
        self.positions = reduce_(self.positions, self.mappings, self.num_pos)
        return self

    def equals(self, other):
        return (other.positions == self.positions).all()

    def to_linear_array(self):
        return self.positions

    def rand_move(self, recursed=0):
        free = (self.positions == 0).nonzero()[0]
        move = np.random.choice(free)
        self.move(move)
        return self

    def greedy_move(self, weights_arr):
        best_score = np.inf * -1
        best_move = 0
        for move in range(self.num_pos):
            if self.positions[move] == 0:
                board = self.move_clone(move)
                reward = board.reward(weights_arr)
                if reward > best_score:
                    best_move = move
                    best_score = reward
        self.move(best_move)
        return self

    def move_available(self, index):
        return self.positions[index] == 0

    def rl_move(self):
        player_symbol = (self.turn % self.q) + 1
        if not self.RL_models[player_symbol]:
            model = DeepQNetwork(alpha=0.003, n_actions=self.num_pos, num_pos=self.num_pos, fc1_dims=2048,
                                 fc2_dims=2048, q=self.q)
            model_name = ''
            model_training = 0
            for mname in os.listdir("./models/"):
                if "games_%dn_%dk_%dq_player%d.pth" % (self.n, self.k, self.q, player_symbol) in mname:
                    mtraining = int(mname[:mname.index("games_")])
                    if mtraining > model_training:
                        model_name = mname
                        model_training = mtraining
            if model_name:
                model.load_state_dict(torch.load("./models/" + model_name))
                model.eval()
                self.RL_models[player_symbol] = model
                print("loaded model", model_name, "with", model_training, "games' experience.")
            else:
                raise LookupError("Cannot find a RL model for the specified n, k, q, playerNo! Train one and save it,"
                                  " then try again.")
        fwded = self.RL_models[player_symbol].forward(self.to_linear_array())
        actions = fwded
        taken = (self.to_linear_array() != 0)  # todo: apply to probabilistic move choice
        actions = actions.masked_fill(torch.tensor(taken, device='cuda'), -np.inf)
        action = torch.argmax(actions).item()
        self.move(action)
        return self

    def reward(self, weights_arr):
        return reward_(self.n, self.k, self.q, self.positions, self.lines, self.num_pos, self.num_lines,
                       weights_arr)

    def playout_plain(self, num_games):
        outcomes = {i: 0 for i in range(self.q + 1)}
        for _ in range(num_games):
            b = copy(self)
            while not b.win():
                b.rand_move()
            winner = b.win()
            if winner == -1:
                winner = 0
            outcomes[winner] += 1
        return outcomes

    def playout(self, num_games):
        res = playout_hyb(self.positions, num_games, self.lines, self.num_pos, self.turn, self.n, self.q)
        return dict(zip(*np.unique(res, return_counts=True)))

    def carlo_greedy(self, count, weights):
        best_score = np.inf * -1
        best_move = 0
        for move in range(self.num_pos):
            if self.positions[move] == 0:
                board = self.move_clone(move)
                outcomes = board.playout(count)
                reward = -1 * sum(outcomes.values())
                # set to 1 to incentivize only drawing
                for outcome in outcomes:
                    reward += outcomes[outcome] * weights[outcome]
                if reward > best_score:
                    best_move = move
                    best_score = reward
        self.move(best_move)
        return self

    def __copy__(self):
        clone_board = Board(self.n, self.k, self.q, self.num_pos, self.positions.copy(), self.lines, self.mappings,
                            self.turn, self.RL_models)
        return clone_board

    def __deepcopy__(self, memo):  # https://stackoverflow.com/a/15774013/12387665
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result

    def __str__(self):
        return "%d**%d Board:" % (self.n, self.k) + str(self.positions)

    def cli(self):
        if self.k == 2:

            # construct a 2D array, then print nicely
            sub = {0: "   ", 1: " X ", 2: " O ", 3: " # ", 4: " + ", 5: " & ", 6: " L "}
            board = list(map(lambda x: sub[x], list(self.positions)))
            board = ''.join(board)
            board_split = [board[i * 3:(i + self.n) * 3] for i in range(0, self.num_pos, self.n)]
            for i in range(self.n):
                board_split[i] = str(i) + board_split[i] + str(i)
            head_foot = ' ' + ''.join([' ' + str(i) + ' ' for i in range(self.n)])
            print(head_foot)
            for row in board_split:
                print(row)
            print(head_foot)
        elif self.k == 3:
            head_foot = '       ' + '       '.join(
                [(''.join([' %1d ' % i for i in range(self.n)])) for _ in range(self.n)])
            level_line = (" " * 2 * self.n) + (" " * (5 * self.n - 5)).join(
                ["Level %1d" % lvl for lvl in range(self.n)])
            d = {0: " ", 1: "X", 2: "O", 3: "#", 4: "+", 5: "&", 6: "L"}
            board = np.array(list(map(lambda x: d[x], self.positions))).reshape([self.n] * 3)
            # reshape to 3D, then print nicely
            boardLines = ["" for _ in range(self.n)]
            for level in range(self.n):
                for row in range(self.n):
                    if level == 0:
                        boardLines[row] += "   %1d   " % row
                    boardLines[row] += " %s " * self.n % tuple(
                        board[level, row])  # leave %1d, makes spacing consistent once replaced
                    if level < self.n:
                        boardLines[row] += "   %1d   " % row
            print(level_line)
            print(head_foot)
            for bL in boardLines:
                print(bL)
            print(head_foot)
            print(level_line)

    def human_move(self):
        if self.k not in [2, 3]:
            raise ValueError("Human move not implemented for dimensionality not equal to 2 or 3.")
        print("The current board state is:")
        self.cli()
        print("This is represented by the array:")
        print(self.positions)
        try:
            level = 0 if self.k == 2 else int(input("Which level would you like to play on (0-%d)? " % (self.n - 1)))
            column = int(input("Which column would you like to play on (0-%d)? " % (self.n - 1)))
            row = int(input("Which row would you like to play on (0-%d)? " % (self.n - 1)))
            choice = level * self.n ** 2 + row * self.n + column
            assert self.move_available(choice)
            assert level in range(self.n)
            assert column in range(self.n)
            assert row in range(self.n)
        except Exception:
            print("Invalid move; perhaps you did not enter integers in the range,"
                  " or input the coordinates of an already-occupied cell.")
            print()
            print("Please try again:")
            self.human_move()
            return
        self.move(choice)
        return self

    def guided_tree(self, guide, player_no, weights, bound=1000):
        """
        Perform guided tree search on the board, starting at the current state.
        :param guide: what algorithm/heuristic chooses moves to guide the tree
        :param player_no:
        """
        startTime = time.time()
        if guide not in ['human', 'random', 'greedy', 'rl', 'carlo']:
            raise ValueError("Tree guide must be one of: 'human'; 'random'; 'greedy'; 'rl'.")
        win_count = {i: 0 for i in range(self.q + 1)}  # win_count[0] counts draws
        leaf_count = 0
        states = {deepcopy(self)}
        children = set()
        depth = 0
        while states:
            depth += 1
            print("At depth: %d; active nodes: %d" % (depth, len(states)))
            for parent in states:
                winner = parent.win()
                if winner:
                    if winner != player_no and winner != -1:
                        print("Player lost:")
                        parent.cli()
                    if winner == -1:  # if a draw
                        win_count[0] += 1
                        leaf_count += 1
                    else:
                        win_count[winner] += 1
                        leaf_count += 1
                else:
                    # produce children; all at unguided transitions, else consult oracle
                    if (parent.turn % parent.q) + 1 == player_no:
                        if guide == 'human':
                            children.update({parent.human_move()})
                        elif guide == 'greedy':
                            children.update({parent.greedy_move()})
                        elif guide == 'random':
                            children.update({parent.rand_move()})
                        elif guide == 'rl':
                            children.update({parent.rl_move()})
                        elif guide == 'carlo':
                            children.update({parent.carlo_greedy(bound, weights)})
                    else:
                        children.update(parent.successors())
            states = children.copy()
            children = set()
            print("After %dth move: %d active nodes" % (depth, len(states)))
            print("Leaf count: {}".format(leaf_count))
            print("Draw count: {}".format(win_count[0]))
            for player in range(1, self.q + 1):
                print("Wins for player {}: {}".format(player, win_count[player]))
                print("Losses for player {}: {}".format(player, leaf_count - win_count[0] - win_count[player]))
            print("\n\n")
        print("Exhausted all configurations in {} seconds.".format(time.time() - startTime))
        print("Leaf count: {}".format(leaf_count))
        print("Draw count: {}".format(win_count[0]))
        for player in range(1, self.q + 1):
            print("Wins for player {}: {}".format(player, win_count[player]))
            print("Losses for player {}: {}".format(player, leaf_count - win_count[0] - win_count[player]))

    def rev_guided_tree(self, guide, player_no, weights, bound=1024):
        """
        Perform guided tree search on the board, starting at the current state.
        :param guide: what algorithm/heuristic chooses moves to guide the tree
        :param player_no:
        """
        startTime = time.time()
        if guide not in ['human', 'random', 'greedy', 'rl', 'carlo']:
            raise ValueError("Tree guide must be one of: 'human'; 'random'; 'greedy'; 'rl'.")
        win_count = {i: 0 for i in range(self.q + 1)}  # win_count[0] counts draws
        leaf_count = 0
        states = {deepcopy(self)}
        children = set()
        depth = 0
        weights_arr = np.array(list(weights.values()))
        while states:
            depth += 1
            print("At depth: %d; active nodes: %d" % (depth, len(states)))
            for parent in states:
                winner = parent.win()
                if winner:
                    if winner == -1:
                        print("Draw:")
                        parent.cli()
                    if winner == -1:  # if a draw
                        win_count[0] += 1
                        leaf_count += 1
                    else:
                        win_count[winner] += 1
                        leaf_count += 1
                else:
                    # produce children; consult oracle for every move except target player
                    if (parent.turn % parent.q) + 1 != player_no:
                        if guide == 'human':
                            children.update({parent.human_move()})
                        elif guide == 'greedy':
                            children.update({parent.greedy_move(weights_arr)})
                        elif guide == 'random':
                            children.update({parent.rand_move()})
                        elif guide == 'rl':
                            children.update({parent.rl_move()})
                        elif guide == 'carlo':
                            children.update({parent.carlo_greedy(bound, weights)})
                    else:
                        children.update(parent.successors())
            states = children.copy()
            children = set()
            print("After %dth move: %d active nodes" % (depth, len(states)))
            print("Leaf count: {}".format(leaf_count))
            print("Draw count: {}".format(win_count[0]))
            for player in range(1, self.q + 1):
                print("Wins for player {}: {}".format(player, win_count[player]))
                print("Losses for player {}: {}".format(player, leaf_count - win_count[0] - win_count[player]))
            print("\n\n")
        print("Exhausted all configurations in {} seconds.".format(time.time() - startTime))
        print("Leaf count: {}".format(leaf_count))
        print("Draw count: {}".format(win_count[0]))
        for player in range(1, self.q + 1):
            print("Wins for player {}: {}".format(player, win_count[player]))
            print("Losses for player {}: {}".format(player, leaf_count - win_count[0] - win_count[player]))


def generate_lines(n, k):
    num_pos = n ** k
    moves = []  # the set of empty moves needs to contain every possible move
    for val in range(num_pos):
        move = []
        # write, for i.e. 3 in base 2: 1,1,0,0...
        # first, write the correct representation in base n
        while val > 0:
            quo = val // n
            rem = val % n
            move.append(rem)
            val = quo
        # then, append zeroes to grow to size
        while len(move) < k:
            move.append(0)
        # no need to reverse, since we don't care about the order the moves are in
        moves.append(move)
    lines = []
    # choose the starting vector.
    for vector0 in moves:
        # Any winning set will necessarily include at least one vector containing at least one zero.
        # Assume wlog that this is vector0
        if 0 not in vector0:
            continue
        # choose a second vector different from the first
        for vector1 in moves:
            if vector0 == vector1:
                continue
            # calculate the "gradient"
            grads = [x1 - x0 for (x0, x1) in zip(vector0, vector1)]
            # print("Gradient vector is:", grads)
            # if the 2 vectors are NOT adjacent, break:
            if not all(map(lambda x: x in [-1, 0, 1], grads)):
                continue
            v_i = vector1
            line = [vector0, vector1]
            for i in range(2, n):  # start at 2, no need to check v_0 and v_1
                v_i = list(map(sum, zip(v_i, grads)))
                line.append(v_i)
            # check if line generated is entirely inside the board
            # i.e. [0,3] and [1,4] as starting point on a 5^2 board, would produce a line that extends outside board
            valid = True
            for i in line[-1]:  # need only check the end of the line - set of moves in board is convex
                if i < 0 or i > n - 1:  # if outside board on any of the coords
                    valid = False
            if valid:
                lines.append(line)
    # eliminate duplicates
    lines = list(map(lambda x: sorted(x), lines))  # represent lines in sorted order
    unique_lines = []
    flattened_lines = []
    for line in lines:
        if line in unique_lines:
            continue
        unique_lines.append(line)
        flattened_lines.append(flatten_line(line, n))
    sorted_lines = list(map(lambda x: list(sorted(x)), flattened_lines))
    return np.array(sorted_lines)


def flatten(point, n):
    total = 0
    for i in range(len(point)):
        total += point[i] * n ** i
    return total


def flatten_line(line, n):
    return list(map(lambda x: flatten(x, n), line))


def generate_transforms(n, k):
    num_pos = n ** k
    base_np = np.reshape(np.arange(num_pos), [n] * k)
    # produce all 'one-step' transforms
    transforms = []
    for dim in range(k):
        # produce base flipped through axis dim
        transforms.append(np.flip(base_np, dim))
        for dim_ in range(k):
            # produce base rotated in the plane given by dim and dim_
            if dim != dim_:
                transforms.append(
                    np.rot90(base_np, k=1, axes=(dim, dim_)))  # no need to use other k, can compose with self
    transforms = [np.reshape(arr, num_pos) for arr in transforms]
    collection = [[i for i in range(num_pos)]]
    collection_grew = True
    while collection_grew:
        collection_grew = False
        for transform in collection:
            for base_transform in transforms:
                temp_tf = apply_transform(transform, base_transform, num_pos)
                if temp_tf not in collection:
                    collection.append(temp_tf)
                    collection_grew = True
    return np.array([np.array(transform) for transform in collection])


def apply_transform(base, transform, num_pos):
    """
    :param num_pos: length of the base and transform arrays
    :param base: 1-D array to be transformed
    :param transform: 1-D transform to apply
    """
    return [base[transform[i]] for i in range(num_pos)]


def fast_transform(base, transform):
    return base[transform]


@jit(nopython=True)
def arr_lt(arr1, arr2, num_pos):
    """
    :param arr1: an n**k integer array
    :param arr2: an n**k integer array
    :return: True iff arr1 is less than arr2
    """
    for i in range(num_pos):
        if arr1[i] < arr2[i]:
            return True
        if arr2[i] < arr1[i]:
            return False
    return False


@jit(nopython=True)
def reduce_(positions, mappings, num_pos):
    best = positions
    for mapping_index in range(mappings.shape[0]):
        mapping = mappings[mapping_index]
        candidate = positions[mapping]
        if arr_lt(candidate, best, num_pos):
            best = candidate
    positions = best
    return positions


@jit(nopython=True)
def reward_(n, k, q, positions, lines, num_pos, num_lines, weights_arr):
    symbolSums = [(positions.take(lines) == i).sum(axis=1, dtype=np.int16) for i in range(q + 1)]
    for player in range(1, q + 1):
        if np.any(symbolSums[player] == n):  # if an opponent is in a winning configuration
            return weights_arr[player] * num_lines * num_pos
    total = 0
    for player in range(1, q + 1):
        player_blockers = [i for i in range(q + 1) if i != 0 and i != player]
        player_excl = np.zeros(num_lines) == 1  # zeros array of dtype boolean, i.e. np.false(num_lines)
        for blocker in player_blockers:
            player_excl = np.logical_or(symbolSums[blocker] != 0, player_excl)
        player_incl = np.logical_and(np.logical_not(player_excl), symbolSums[player])
        total += (weights_arr[player] / (n - symbolSums[player][player_incl])).sum()
    return total
    """ comment in report - enforcing the game is zero-sum yields quite poor play"""


# return Q if player Q has a winning line (if more than one player has a winning line, the lowest-number player is
# returned); returns 0 if not a winning configuration
@jit(nopython=True)
def win_(n, k, q, positions, lines, num_pos, num_lines):
    symbolSums = [(positions.take(lines) == i).sum(axis=1, dtype=np.int16) for i in range(q + 1)]
    for player in range(1, q + 1):
        if np.any(symbolSums[player] == n):  # if player controls n cells in a row
            return player
    if (positions != 0).all():  # if board is full, draw
        return -1
    return 0


@jit(nopython=True)
def playout_ind(positions, num_games, lines, num_pos, turn, n, q):
    # make num_games copies of board in a 2D array
    positions.astype(np.int16)
    wins = np.zeros(num_games)  # 0 if active, Q if won by player Q
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
    out = np.empty((num_batches, 128), dtype=np.int16)
    for index in range(num_batches):
        out[index] = playout_ind(positions, 128, lines, num_pos, turn, n, q)
    return out


def rev_tree_carlo(n, k, q, lines, mappings, start_board, coalition_target, weights, mc_num_games):
    startTime = time.time()
    turn = (start_board != 0).sum()  # turn is the number of moves made so far
    num_pos = n ** k
    num_lines = len(lines)
    win_count = {i: 0 for i in range(q + 1)}  # win_count[0] counts draws
    leaf_count = 0
    states = {tuple(start_board.copy())}  # use tuples - immutable, implement hashing for set functionality
    children = set()
    depth = 0
    while states:
        depth += 1
        print("At depth: %d; active nodes: %d" % (depth, len(states)))
        for parent in states:
            np_parent = np.array(parent, dtype=np.int16)
            winner = win_(n, k, q, np_parent, lines, num_pos, num_lines)
            if winner:
                leaf_count += 1
                if winner == -1:  # if a draw
                    win_count[0] += 1
                    print("Draw:", np_parent)
                else:
                    if winner == coalition_target:
                        print("Target win:", np_parent)
                    win_count[winner] += 1
            else:
                # produce children; consult oracle for every move except target player
                player = (turn % q) + 1  # todo: turn tracking
                if player != coalition_target:
                    best_score = np.inf * -1
                    best_move = 0
                    possible_moves = (np_parent == 0).nonzero()[0]
                    for move in possible_moves:
                        np_parent[move] = player  # do in-place work on np_parent to improve runtime
                        playout = playout_hyb(np_parent, mc_num_games, lines, num_pos, turn + 1, n, q)
                        outcomes = dict(zip(*np.unique(playout, return_counts=True)))
                        reward = 0
                        for outcome in outcomes:
                            reward += outcomes[outcome] * weights[outcome]
                        if reward > best_score:
                            best_move = move
                            best_score = reward
                        np_parent[move] = 0  # revert parent
                    np_parent[best_move] = player
                    children.add(tuple(reduce_(np_parent, mappings, num_pos)))
                else:
                    possible_moves = (np_parent == 0).nonzero()[0]
                    for move in possible_moves:
                        np_parent[move] = player  # do in-place work on np_parent to improve runtime
                        children.add(tuple(reduce_(np_parent, mappings, num_pos)))
                        np_parent[move] = 0  # revert to parent state

        states = children
        children = set()
        turn += 1
        print("After %dth move: %d active nodes; computing for %3f seconds" % (
            depth, len(states), time.time() - startTime))
        print("Leaf count: {}".format(leaf_count))
        print("Draw count: {}".format(win_count[0]))
        for player in range(1, q + 1):
            print("Wins for player {}: {}".format(player, win_count[player]))
            print("Losses for player {}: {}".format(player, leaf_count - win_count[0] - win_count[player]))
        print("\n\n")
    print("Exhausted all configurations in {} seconds.".format(time.time() - startTime))
    print("Leaf count: {}".format(leaf_count))
    print("Draw count: {}".format(win_count[0]))
    for player in range(1, q + 1):
        print("Wins for player {}: {}".format(player, win_count[player]))
        print("Losses for player {}: {}".format(player, leaf_count - win_count[0] - win_count[player]))

