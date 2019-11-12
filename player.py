from abc import ABC, abstractmethod


class Player(ABC):
    def __init__(self, k, n, q):
        self.k = k
        self.n = n
        self.q = q
        self.name = ""

    @abstractmethod
    def move(self, O, X, E, turn):
        pass

    def name(self):
        return self.name