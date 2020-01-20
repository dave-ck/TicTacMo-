from abc import ABC, abstractmethod
import random


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

    def reward(self, config):   # called on game wins
        pass

    def punish(self, config):   # called on game losses
        pass

    def draw(self, config):     # called on game draws
        pass

class Randy(Player):
    def __init__(self, k, n, q):
        super().__init__(k, n, q)
        self.name = "Randy"

    def move(self, O, X, E, turn):  # refactor to take "config" as input
        # picks a random member of E and returns it as the "expert" move
        move = random.sample(E, 1)[0]
        # print("Randy plays randomly: chooses {} from {}".format(move, E))
        return move

    def reward(self, config):
        print("Win!")
        config.print_2d()
        print("\n\n")

    def punish(self, config):
        print("Loss :(")
        config.print_2d()
        print("\n\n")

    def draw(self, config):
        print("Draw :/")
        config.print_2d()
        print("\n\n")
