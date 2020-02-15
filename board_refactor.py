import time
import random
import numpy as np
from copy import deepcopy

class Board:  # Cimpl entire class as a struct, functions as methods taking the struct as a parameter

    def __init__(self, n, k, num_pos, positions, lines, turn):
        self.name = "Game"
        self.n = n  # number of
        self.k = k  # number of dimensions
        self.num_pos = num_pos  # precomputed and passed as param for speed, p = n**k
        self.turn = turn  # specific to configuration
        self.lines = lines or generate_lines(n, k)
        self.positions = positions  # 0 for empty, 1 for X (first and odd moves), -1 for O (second and even moves)

    @classmethod
    def blank_board(cls, n, k):
        num_pos = n ** k
        positions = [0 for _ in range(num_pos)]
        lines = generate_lines(n, k)
        return cls(n, k, num_pos, positions, lines, turn=0)

    def reset(self):
        self.turn = 0
        positions = [0 for _ in range(self.num_pos)]

    def successors(self):  # Cimpl with isomorphism checks
        """
        Produces all possible successors of the Board configuration (which are 1 move away). Does not reduce these.
        :return: A set containing Board objects.
        """
        successors = set()
        for index in range(self.num_pos):
            if self.positions[index] == 0:
                successors.update({self.move_clone(self.positions[index])})
        return successors

    # consider refactoring to only consider whether the *last* move is a part of a win
    def win(self, player):
        for line in self.lines:
            line_win = True
            for index in line:
                line_win = line_win and self.positions[index] == player
            if line_win:
                return True
        return False

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
        return 0 not in self.positions

    def to_linear_array(self):
        return np.array(self.positions, dtype='int8')

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
    return flattened_lines


def flatten(point, n):
    total = 0
    for i in range(len(point)):
        total += point[i]*n**i
    return total


def flatten_line(line, n):
    return list(map(lambda x: flatten(x, n), line))


b = Board.blank_board(3, 2)
b.print_2d()
print(b.lines)
print(b.positions)

time.sleep(1)
while not b.draw():
    print("Move")
    b.rand_move()
    b.print_2d()
