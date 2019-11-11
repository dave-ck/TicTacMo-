from win_check import win
from config import Config


class Game:
    def __init__(self, q, n, k, expert, i=0):
        """
        @:param q: the number of players
        @:param n: the side length of the board
        @:param k: the number of dimensions the board extends in
        @:param expert: pass an instance of the player class to give moves assumed to be perfect play
        @:param i: which place the expert plays in. 0 if the expert plays first, q-1 if the expert plays last.
        """
        self.name = "Game"
        self.q = q
        self.n = n
        self.k = k
        self.p = n ** k  # precompute for speed
        self.expert = expert
        self.i = i
        # initialize set of configurations with just the empty configuration
        self.configs = {Config(self.n, self.k, self.p, set(), set(), self.enumerate_moves())}
        self.successors = set()
        self.turn = 0
        self.winnable = True  # update with logical ANDs - reflects whether player i can always win with perfect play
        self.drawable = True  # update with logical ANDs - reflects whether player i can always draw with perfect play
        self.computed = False  # flag to signal whether or not the vars winnable and drawable mean anything

    def play(self):
        print("Waddup")
        # for every config
        for config in self.configs:
            if config.win():
                if self.turn % self.q == self.i:  # if player i made the latest move, i.e. won the game
                    self.winnable = True and self.winnable  # do not overwrite if false
                    # no need to update self.drawable; if it is at any point False, we interrupt play
                else:  # player i did not make the latest move; our "expert" failed to play perfectly, or the game is
                    # neither winnable nor drawable with perfect play by player i
                    self.winnable = False
                    self.drawable = False
                    return
            elif self.turn == self.p:
                self.winnable = False
                # no need to execute: self.drawable = True and self.drawable
            else:
                self.successors.update(config.successors()) # add all of the configuration's successors to the set
        # omitted: identify and eliminate isomorphic boards among successors
        self.configs = self.successors
        self.successors = set()
        if self.configs:    # if there remain any configurations to expand, then play another turn
            self.play()

    def enumerate_moves(self):  # Cimpl
        # return the n**m possible moves
        # generate from value, converted to base m
        moves = []
        for val in range(self.p):
            move = []
            # write, for i.e. 3 in base 2: 1,1,0,0...
            # first, write the correct representation in base n
            while val > 0:
                quo = val // self.n
                rem = val % self.n
                move.append(rem)
                val = quo
            # then, append zeroes to grow to size
            while len(move) < self.k:
                move.append(0)
            # no need to reverse, since we don't care about the order the moves are in
            moves.append(move)
        return set(moves)


classic = Game(2, 3, 2)
classic.play()
