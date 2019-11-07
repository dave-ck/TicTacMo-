import random
from player import Player


class Randy(Player):
    def __init__(self, k, n, q):
        self.__init__(self, k, n, q)

    def move(self, movesO, movesX, xMove=True):
        # create a random k-vector, not in either set O or X, with each value in the interval [0, n]
        vector = [random.randint(0, self.n) for i in range(self.k)]
        if vector not in movesO and vector not in movesX:
            return vector
        else:
            return self.move(movesO, movesX, xMove)
