import random
from player import Player


class Randy(Player):
    def __init__(self, k, n, q):
        super().__init__(k, n, q)
        self.name = "Randy"

    def move(self, O, X, E, turn):
        # picks a random member of E and returns it as the "expert" move
        move = random.sample(E, 1)[0]
        print("Randy plays randomly: chooses {} from {}".format(move, E))
        return move
