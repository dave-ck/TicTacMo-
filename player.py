from abc import ABC, abstractmethod


class Player(ABC):
    def __init__(self, k, n, q):
        self.k = k
        self.n = n
        self.q = q

    @abstractmethod
    def move(self, movesO, movesX, xMove=True):
        pass
