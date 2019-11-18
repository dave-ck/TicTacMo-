from win_check import win


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
        # if expert isn't playing
        # for every possible smove
        # winnable = winnable and winnable(configuration resulting from move)

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
        successor_O = self.O.copy()
        successor_X = self.X.copy()
        if self.turn % 2:  # if O is about to move
            successor_O.append(position)
        else:  # X is about to move
            successor_X.append(position)
        successor_E = self.E.copy()
        successor_E.remove(position)  # mindful of removing list (object) from set...
        return Config(self.n, self.k, self.p, successor_O, successor_X, successor_E, self.turn+1)

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
