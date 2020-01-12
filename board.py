import random
import numpy as np
from win_check import win
from copy import deepcopy


class Board:  # Cimpl entire class as a struct, functions as methods taking the struct as a parameter

    def __init__(self, n, k, num_pos, O, X, E, turn):
        self.name = "Game"
        self.n = n  # number of
        self.k = k  # number of dimensions
        self.num_pos = num_pos  # precomputed and passed as param for speed, p = n**k
        self.O = O  # set of O-positions
        self.X = X  # set of X-positions
        self.E = E  # set of empty positions; equal to the set of all moves with O and X removed
        self.turn = turn  # specific to configuration

    @classmethod
    def blank_board(cls, n, k):
        num_pos = n**k
        E = [] # the set of empty moves needs to contain every possible move
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
            E.append(move)
        return cls(n, k, num_pos, [], [], E, 0)

    def successors(self):  # Cimpl with isomorphism checks
        """
        Produces all possible successors of the Board configuration (which are 1 move away). Does not reduce these.
        :return: A set containing Board objects.
        """
        successors = set()
        for position in self.E:
            successors.update({self.move_clone(position)})
        return successors

    def win(self):  # may refactor for speed; don't recompute win(X) and win(O) later by return reason why terminal
        if win(self.O, self.n):
            return True
        elif win(self.X, self.n):
            return True
        return False

    def move(self, position):  # return a copy of the config, with the move added
        if position not in self.E:  # illegal move; does not affect game state and player loses their move
            # print("Illegal move suppressed.")
            return
        if self.turn % 2:  # if O is about to move
            self.O.append(position)
        else:  # X is about to move
            self.X.append(position)
        self.E.remove(position)  # mindful of removing list (object) from set...
        self.turn += 1

    def move_clone(self, position):
        """
        Clones the current Board, makes the move on the clone, then returns the clone.
        :param position: The position, as a list, where the move should be made (i.e. [0,0] for the top left, in 2D)
        :return: the cloned Board with the move applied.
        """
        clone_board = deepcopy(self)
        clone_board.move(position)
        return clone_board


    def draw(self):
        if len(self.X) + len(self.O) == self.num_pos:
            return True
        return False

    def to_linear_array(self):
        # -1 indicates an X, 0 indicates empty, 1 indicates an O
        board = np.zeros([self.n] * self.k, dtype='int8')  # initialize empty board; for Cimpl: need 2 bits per cell
        for move in self.O:
            board[tuple(move)] = 1
        for move in self.X:
            board[tuple(move)] = -1
        return board.flatten()

    def rand_move(self):
        move = random.sample(self.E, 1)[0]
        self.move(move)

    def move_available(self, position):
        return position in self.E

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
        board = [["  " for _ in range(self.n)] for _ in range(self.n)]
        for x in self.O:
            x, y = x[0], x[1]
            board[y][x] = "O "
        for x in self.X:
            a, b = x[0], x[1]
            board[b][a] = "X "
        # print nicely
        row_num = 0
        print("  " + "".join(list(str(i) + " " for i in range(self.n))))
        for row in board:
            row = str(row_num) + " " + "".join(row) + str(row_num)
            row_num += 1
            print(row)
        print("  " + "".join(list(str(i) + " " for i in range(self.n))))

# c = Config(3,2,3**2, [], [], [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]])
# for testing
