from win_check import win


class Config:  # Cimpl entire class as a struct, functions as methods taking the struct as a parameter

    def __init__(self, n, k, p, O, X, E):
        self.name = "Game"
        self.O = O
        self.X = X
        self.E = E
        self.n = n  # number of
        self.k = k  # number of dimensions
        self.p = p  # precomputed and passed as param for speed, p = n**k
        self.O = set()  # set of O-positions
        self.X = set()  # set of X-positions
        self.E = set()  # set of empty positions; equal to the set of all moves with O and X removed
        self.turn = 0  # specific to configuration
        # if expert isn't playing
        # for every possible smove
        # winnable = winnable and winnable(configuration resulting from move)

    def successors(self):  # Cimpl with isomorphism checks
        successors = set()
        for move in self.E:
            successor_O = self.O.copy()
            successor_X = self.X.copy()
            if self.turn % 2:  # if O is about to move; consider all possible moves for O
                successor_O.update({move})
            else:  # X is about to move; consider all possible moves for X
                successor_X.update({move})
            successor_E = self.E.copy()
            successor_E.remove(move)  # mindful of removing list (object) from set...
            successors.update({Config(self.n, self.k, self.p, successor_O, successor_X, successor_E)})
        return successors

    def win(self):  # may refactor for speed; don't recompute win(X) and win(O) later by return reason why terminal
        if win(self.O, self.n):
            return True
        elif win(self.X, self.n):
            return True
        return False

    def draw(self):
        if len(self.X) + len(self.O) == self.p:
            return True
        return False
