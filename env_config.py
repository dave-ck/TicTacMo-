import random
import numpy as np
from win_check import win
from copy import deepcopy


class Config:  # Cimpl entire class as a struct, functions as methods taking the struct as a parameter

    def __init__(self, n, k, p, O, X, E, turn=0):
        self.name = "Game"
        self.n = n  # number of
        self.k = k  # number of dimensions
        self.p = p  # precomputed and passed as param for speed, p = n**k
        self.O = O  # set of O-positions
        self.X = X  # set of X-positions
        self.E = E  # set of empty positions; equal to the set of all moves with O and X removed
        self.turn = turn  # specific to configuration

    def successors(self):  # Cimpl with isomorphism checks
        successors = set()
        for position in self.E:
            successors.update({self.move(position)})
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

    def draw(self):
        if len(self.X) + len(self.O) == self.p:
            return True
        return False

    def print_2d(self):
        if self.k != 2:
            print("Can't print a game with k != 2!")
            return
        # construct a 2D array, then print nicely
        board = [[" " for _ in range(self.n)] for _ in range(self.n)]
        for x in self.O:
            x, y = x[0], x[1]
            board[y][x] = "O"
        for x in self.X:
            a, b = x[0], x[1]
            board[b][a] = "X"
        # print nicely
        for row in board:
            row = "".join(row)
            print(row)
        print("_" * self.n)

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
