import random
import numpy as np
from win_check import win
from copy import deepcopy


# todo: double-check reward function is working correctly


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

    def reward(self, q, win_forcer=-1):
        """
        Generate rewards for each player in the game at current time. Punishes players who can force a win for drawing.
        Punishes players who can force a draw for losing.
        :param q: the total number of players
        :param win_forcer: number of the player who can force a win, or -1 if no such player
        :return: a dictionary {player_number: reward}
        """
        latest_player = (self.turn - 1) % q
        if win(self.X, self.n) or win(self.O, self.n):
            # heaviest punishment is for losses close to the beginning
            # i.e. losing after 5 turns on a 3*3 board: reward of -9-1+5=-5
            # i.e. losing after 9 turns on a 3*3 board: reward of -9-1+9=-1
            # i.e. winning after 5 turns on a 3*3 board: reward of 9+1-5= 5
            # i.e. winning after 9 turns on a 3*3 board: reward of 9+1-9= 1
            rewards = {i: (-1 * self.p) - 1 + self.turn for i in
                       range(q)}  # heavy punishment for losing; if inevitable, doesn't matter
            if win_forcer != -1:
                rewards[win_forcer] *= 2  # punish the win forcer more if they lost
            rewards[latest_player] = self.p + 1 - self.turn  # reward the winner
        elif self.draw():
            rewards = {i: 0 for i in range(q)}
            if win_forcer in range(q):
                # draw is always at the last move; function of self.p to keep relevant with large p
                rewards[win_forcer] = self.p * -0.5
        else:
            rewards = {i: 0 for i in range(q)}  # should not need to be called, but may be
        return rewards

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
        self.turn += 1

    def draw(self):
        if len(self.X) + len(self.O) == self.p:
            return True
        return False

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

#
# c = Config(3, 2, 3 ** 2, [], [], [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]])
# # for testing
# c.move([1, 2])
# c.move([2, 0])
# c.move([1, 1])
# c.move([2, 1])
# c.move([1, 0])
# c.print_2d()
# print(c.reward(2))
