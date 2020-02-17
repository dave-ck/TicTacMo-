import time
import random
import numpy as np
from copy import deepcopy
import ctypes

#############
## C setup ##
#############
lib = ctypes.cdll.LoadLibrary('./cimpl/cboard.so')
cdraw = lib.draw
cwin = lib.win
init_lines = lib.initLines
init_vars = lib.initVars
c_print = lib.printArr
print_lines = lib.printLines

k = 2
n = 3

state_neutral = np.array([0, 0, 0,
                          0, 1, 0,
                          0, 0, -1], dtype='int8')

state_win = np.array([0, -1, 0,
                      1, -1, 1,
                      0, -1, 1], dtype='int8')

state_draw = np.array([1, -1, 1,
                       -1, -1, 1,
                       1, 1, -1], dtype='int8')



print(state_neutral)
print("Draw:", cdraw(ctypes.c_void_p(state_neutral.ctypes.data)))
print("Win 1:", cwin(ctypes.c_void_p(state_neutral.ctypes.data), ctypes.c_int8(1)))
print("Win -1:", cwin(ctypes.c_void_p(state_neutral.ctypes.data), ctypes.c_int8(-1)))
c_print(ctypes.c_void_p(state_neutral.ctypes.data))



##############
## /C setup ##
##############



class Board:  # Cimpl entire class as a struct, functions as methods taking the struct as a parameter

    def __init__(self, n, k, num_pos, positions, lines, turn):
        self.name = "Game"
        self.n = n  # number of
        self.k = k  # number of dimensions
        self.num_pos = num_pos  # precomputed and passed as param for speed, p = n**k
        self.turn = turn  # specific to configuration
        self.lines = lines
        self.positions = positions  # 0 for empty, 1 for X (first and odd moves), -1 for O (second and even moves)

    @classmethod
    def blank_board(cls, n, k):
        num_pos = n ** k
        positions = np.zeros([num_pos], dtype='int8')
        init_vars(n, k)
        lines = generate_lines(n, k)    # implement some checking; init_lines and init_vars iff current n, k undesired
        init_lines(ctypes.c_void_p(lines.ctypes.data), ctypes.c_int(lines.shape[0]), ctypes.c_int(lines.shape[1]))
        return cls(n, k, num_pos, positions, lines, turn=0)

    def reset(self):
        self.turn = 0
        self.positions = np.zeros([self.num_pos], dtype='int8')

    def successors(self):  # Cimpl with isomorphism checks
        """
        Produces all possible successors of the Board configuration (which are 1 move away). Does not reduce these.
        :return: A set containing Board objects.
        """
        successors = set()
        for index in range(self.num_pos):
            if self.positions[index] == 0:
                successors.update({self.move_clone(index)})
        return successors

    # consider refactoring to only consider whether the *last* move is a part of a win
    def win(self, symbol):
        return cwin(ctypes.c_void_p(self.positions.ctypes.data), ctypes.c_int8(symbol))


    # todo: rework reward function to return int instead of dict
    def reward(self, player, q, win_forcer=-1):
        """
        Generate a reward for the player in a q-player game based on how "desirable" the current configuration is to
        that player.
        :param player: the player's number - 0 for the 1st player.
        :param q: the total number of players
        :param win_forcer: number of the player who can force a win, or -1 if no such player
        :return: a dictionary {player_number: reward}
        """
        raise NotImplementedError

    def move(self, position):  # return a copy of the config, with the move added
        if self.positions[position] != 0:  # illegal move; does not affect game state and player loses their move
            return
        elif self.turn % 2:  # if O is about to move
            self.positions[position] = -1
        else:  # X is about to move
            self.positions[position] = 1
        self.turn += 1

    def move_clone(self, position):
        """
        Clones the current Board, makes the move on the clone, then returns the clone.
        :param position: The position, as an integer, where the move should be made (i.e. 0 for top left corner, in 2D)
        :return: the cloned Board with the move applied.
        """
        clone_board = deepcopy(self)
        clone_board.move(position)
        return clone_board

    def draw(self):
        return cdraw(ctypes.c_void_p(self.positions.ctypes.data))

    def to_linear_array(self):
        return self.positions

    def rand_move(self, recursed=0):
        if recursed < 100:
            index = random.randint(0, self.num_pos - 1)
            if self.positions[index] == 0:
                self.move(index)
            else:
                self.rand_move(recursed + 1)
        else:
            for index in range(self.num_pos):
                if self.positions[index] == 0:
                    return index
            raise ValueError("No random move is possible on a full board.")

    def move_available(self, index):
        return self.positions[index] == 0

    def __copy__(self):  # https://stackoverflow.com/a/15774013/12387665
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo):  # https://stackoverflow.com/a/15774013/12387665
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result

    def print_2d(self):
        if self.k != 2:
            print("Can't print a game with k != 2!")
            return
        # construct a 2D array, then print nicely
        sub = {-1: " O ", 0: "   ", 1: " X "}
        board = list(map(lambda x: sub[x], self.positions))
        board = ''.join(board)
        board_split = [board[i*3:(i + self.n)*3] for i in range(0, self.num_pos, self.n)]
        for i in range(self.n):
            board_split[i] = str(i) + board_split[i] + str(i)
        head_foot = ' ' + ''.join([' ' + str(i) + ' ' for i in range(self.n)])
        print(head_foot)
        for row in board_split:
            print(row)
        print(head_foot)


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
    ret = np.array(flattened_lines, dtype='int8')
    return ret


def flatten(point, n):
    total = 0
    for i in range(len(point)):
        total += point[i]*n**i
    return total


def flatten_line(line, n):
    return list(map(lambda x: flatten(x, n), line))



class Game:
    def __init__(self, k, n, q, players):
        """
        @:param q: the number of players
        @:param n: the side length of the board
        @:param k: the number of dimensions the board extends in
        @:param players: pass an instance of the player class to give moves assumed to be perfect play,
        or None to have non-deterministic moves. Will be truncated/ padded with None to be length q
        e.g. q=3, players = [None, None, Randy(k, n, q)]
        """
        self.name = "Game"
        self.q = q
        self.n = n
        self.k = k
        self.p = n ** k  # precompute for speed
        self.players = players[:q] + [None] * (
                    q - len(players))  # pad with None (nondeterministic) players up to q size
        # initialize set of configurations with just the empty configuration
        self.fringe = {Board.blank_board(self.n, self.k)}
        self.successors = set()
        self.turn = 0
        self.leaf_count = 0  # keep track of total games ended
        self.draw_count = 0  # keep track of total draws to save computing through summation/subtraction at end
        self.player_wins = {i: 0 for i in range(q)}  # keep track of how many wins each player (play position) has
        self.initTime = time.time()
        print("\n\nBegan processing board with n={}, k={}, q={}".format(self.n, self.k, self.q))

    def play(self):
        # for every config
        print("Looping in play(); {}s since init; turn {}; {} configs at present".format(time.time() - self.initTime,
                                                                                         self.turn, len(self.fringe)))
        for board in self.fringe:
            if board.win(-1) or board.win(1):
                self.leaf_count += 1
                self.player_wins[(self.turn - 1) % self.q] += 1
            elif board.draw():  # board is full; number of turns elapsed == total number of positions
                self.leaf_count += 1
                self.draw_count += 1
            else:
                player = self.players[self.turn % self.q]
                if player:
                    player_move = player.move(board.positions, self.turn)
                    resulting_config = board.move_clone(player_move)
                    self.successors.update({resulting_config})
                else:  # branch to all possible moves
                    self.successors.update(board.successors())  # add all of the configuration's successors to the set
        # omitted: identify and eliminate equivalent boards among successors
        self.fringe = self.successors
        self.successors = set()
        self.turn += 1
        if self.fringe:  # if there remain any configurations to expand, then play another turn
            self.play()
        else:
            print("Exhausted all configurations; {}s since init.".format(time.time() - self.initTime))
            print("Leaf count: {}".format(self.leaf_count))
            print("Draw count: {}".format(self.draw_count))
            for i in range(self.q):
                player = self.players[i]
                if player:
                    print("Wins for player {} ({}): {}".format(i, player.name, self.player_wins[i]))
                    print("Losses for player {} ({}): {}".format(i, player.name,
                                                                 self.leaf_count - self.draw_count - self.player_wins[
                                                                     i]))
                else:
                    print("Wins for player {} (None): {}".format(i, self.player_wins[i]))
                    print("Losses for player {} (None): {}".format(i,
                                                                   self.leaf_count - self.draw_count - self.player_wins[
                                                                       i]))
            # also include print statement to specify whether category membership for values k, n, q has been proven


g = Game(2, 3, 2, [None, None])
g.play()
#
#
# b = Board.blank_board(3, 2)
# b.print_2d()
# print(b.lines)
# print(b.positions)
#
# for i in range(3):
#     print("Move")
#     b.rand_move()
#     b.print_2d()
#
# i = 0
# print("\n\nProducing successors\n\n")
# b.successors()
#
