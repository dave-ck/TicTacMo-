from abc import ABC, abstractmethod


class Player(ABC):
    def __init__(self, k, n, q):
        self.k = k
        self.n = n
        self.q = q
        self.name = ""

    @abstractmethod
    def move(self, O, X, E, turn):  # refactor to take only a "config" object as argument
        pass

    def name(self):
        return self.name

    def reward(self):   # called on game wins
        pass

    def punish(self):   # called on game losses
        pass

    def draw(self):     # called on game draws
        pass