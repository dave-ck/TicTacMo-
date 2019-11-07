from win_check import win

class Game:
    def __init__(self, q, n, k, expertNo=1):
        self.name = "Game"
        self.q = q
        self.n = n
        self.k = k
        self.O = set()
        self.X = set()
        self.turn = 0

    def play(self):
        print("Waddup")


classic = Game(2, 3, 2)
classic.play()