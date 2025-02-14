import random
import numpy as np
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
        num_pos = n ** k
        E = []  # the set of empty moves needs to contain every possible move
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

    def reset(self):
        self.turn = 0
        # may be faster to produce blank board from scratch depending on Cimpl (or maintain set of all moves in mem)
        self.E.extend(self.O)
        self.E.extend(self.X)
        self.X = []
        self.O = []

    def successors(self):  # Cimpl with isomorphism checks
        """
        Produces all possible successors of the Board configuration (which are 1 move away). Does not reduce these.
        :return: A set containing Board objects.
        """
        successors = set()
        for position in self.E:
            successors.update({self.move_clone(position)})
        return successors

    # consider refactoring to only consider whether the *last* move is a part of a win
    def win(self):  # may refactor for speed; don't recompute win(X) and win(O) later by return reason why terminal
        if win(self.O, self.n):
            return True
        elif win(self.X, self.n):
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

"""
We rely on the property, wlog, that any win will have ONE "minimal" vector v_0 -
That is, for every solution V, there is some v_0 which contains at least one 0 (say at index i), such that:
    no other v_j contains a 0 at index i
    every v_k in V contains k at index i (we'll call index i our INCREMENTATION INDEX (name is a work in progress, we'll find something catchier))
"""


def win_imperative(moves_list, n):
    print("Trying {}, {}".format(moves_list, n))
    if not moves_list:
        return False
    moves_list = sorted(moves_list)
    # choose the starting vector.
    for vector0 in moves_list:
        # Any winning set will necessarily include at least one vector containing at least one zero.
        # Assume wlog that this is vector1.
        if 0 not in vector0:
            continue
        # choose a second vector different from the first
        for vector1 in moves_list:
            if vector0 == vector1:
                continue
            # print("Trying pair", vector0, ",", vector1)
            # calculate the "gradient"
            grads = []
            for i in range(len(vector0)):  # same len as vector1 - may need error checking
                grads.append(vector1[i] - vector0[i])
            # if the 2 vectors are NOT adjacent, break:
            adjacent = True
            for grad in grads:
                adjacent = adjacent and grad in [-1, 0, 1]
            if not adjacent:
                continue
            # Compute each v_i which would be in the solution V
            v_i = vector0[:]
            i = 0
            while v_i in moves_list:
                i += 1
                # compute next v_(i+1) by summing v_i and grads pairwise
                for j in range(len(v_i)):  # probably good to just define vector length n as a parameter at func start
                    v_i[j] += grads[j]  # increment current grid cell by grads
            if i == n:  # if the while-loop's condition evaluated to "True" n times
                return True
    return False


def win(moves_list, areallylongfuckingstringgoodgrief):
    moves_list = sorted(moves_list)
    # choose the starting vector.
    for vector0 in moves_list:
        # Any winning set will necessarily include at least one vector containing at least one zero.
        # Assume wlog that this is vector1.
        if 0 not in vector0:
            continue
        # choose a second vector different from the first
        for vector1 in moves_list:
            if vector0 == vector1:
                continue
            # print("Trying pair", vector0, ",", vector1)
            # calculate the "gradient"
            grads = [x1 - x0 for (x0, x1) in zip(vector0, vector1)]
            # print("Gradient vector is:", grads)
            # if the 2 vectors are NOT adjacent, break:
            if not all(map(lambda x: x in [-1, 0, 1], grads)):
                # print("Not adjacent")
                continue
            pairing = True  # Assume vector1, vector2 belong to some solution V
            # Compute each v_i which would be in the solution V
            v_i = vector1
            for i in range(2, areallylongfuckingstringgoodgrief):  # start at 2, no need to check v_0 and v_1
                v_i = list(map(sum, zip(v_i, grads)))
                if v_i not in moves_list:
                    # print(v_i, "was absent from the movesList")
                    pairing = False
                    break  # no need to continue with loop
                    # (note for refactor: can use while-loop, but will probably look clunkier)
                # print(v_i, "in solution")
            if pairing:
                return True
    return False


##############################################################################
#                              TEST CODE BELOW                               #
##############################################################################

"""

win_imperativeTest0 = (
    [[1, 0],  # vanilla win, vertical/horizontal
     [0, 0],
     [2, 0]],
    3)

win_imperativeTest1 = (
    [[1, 1],  # vanilla win, diagonal
     [0, 0],
     [2, 2]],
    3)

win_imperativeTest2 = (
    [[1, 1],
     [0, 0],
     [2, 1],  # spoiler
     [2, 2]],
    3)

win_imperativeTestHighDim = (
    [[1, 1, 2, 2, 1],
     [0, 0, 1, 1, 1],
     [2, 1, 1, 1, 0],
     [2, 2, 2, 2, 0],
     [2, 0, 0, 0, 0]],
    3)

win_imperativeTestBigBoard = (
    [[i, 1] for i in range(20)] + [[i % 7 + 1, i + 2] for i in range(23)] + [[i, 1] for i in range(21, 40)] + [[20, 1]],
    40)




loseTest = (
    [[1, 2],
     [0, 0],
     [1, 1],
     [2, 0]],
    3)

print("win_imperative tests")
print("\n")
print(win_imperative(*win_imperativeTest0), "yeet")
print("\n")
print(win_imperative(*win_imperativeTest1))
print("\n")
print(win_imperative(*win_imperativeTest2))
print("\n")
print("\n")
print("Lose tests")
print("\n")
print(win_imperative(*loseTest))
print("\n")
print("\n")
print("5D - win_imperative")
print("\n")
print(win_imperative(*win_imperativeTestHighDim))

print("\n")
print("\n")
print("High Size, 2D - win_imperative")
print("\n")
print(win_imperative(*win_imperativeTestBigBoard))



winTest0 = (
    [[1,0], #vanilla win, vertical/horizontal
     [0,0],
     [2,0]],
    3)


winTest1 = (
    [[1,1], #vanilla win, diagonal
     [0,0],
     [2,2]],
    3)

winTest2 = (
    [[1,1],
     [0,0],
     [2,1], #spoiler
     [2,2]],
    3)

winTestHighDim = (
    [[1,1,2,2,1],
     [0,0,1,1,1],
     [2,1,1,1,0],
     [2,2,2,2,0],
     [2,0,0,0,0]],
    3)

winTestBigBoard = (
    [[i,1] for i in range(20)]+[[i%7+1, i+2] for i in range(23)]+[[i,1] for i in range(21,40)],
    40)

loseTest = (
    [[1,2],
     [0,0],
     [1,1], 
     [2,0]],
    3)



print("Win tests")
print("\n")
print(win(*winTest0))
print("\n")
print(win(*winTest1))
print("\n")
print(win(*winTest2))
print("\n")
print("\n")
print("Lose tests")
print("\n")
print(win(*loseTest))
print("\n")
print("\n")
print("5D - win")
print("\n")
print(win(*winTestHighDim))

print("\n")
print("\n")
print("High Size, 2D - win")
print("\n")
print(win(*winTestBigBoard))
"""

