from config import Config


class Game:
    def __init__(self, k, n, q, players):
        """
        @:param q: the number of players
        @:param n: the side length of the board
        @:param k: the number of dimensions the board extends in
        @:param players: pass an instance of the player class to give moves assumed to be perfect play,
        or None to have non-deterministic moves. Will be truncated/ padded with None to be length q
        e.g. q=3, players = [None, None, Randy(k, n, q)]
        """
        self.name = "Game"
        self.q = q
        self.n = n
        self.k = k
        self.p = n ** k  # precompute for speed
        self.players = players[:q] + [None] * (q - len(players))
        # initialize set of configurations with just the empty configuration
        self.configs = {Config(self.n, self.k, self.p, [], [], self.enumerate_moves())}
        self.successors = set()
        self.turn = 0
        self.leaf_count = 0  # keep track of total games ended
        self.draw_count = 0  # keep track of total draws to save computing through summation/subtraction at end
        self.player_wins = {i: 0 for i in range(q)}  # keep track of how many wins each player (play position) has

    def play(self):
        # for every config
        print("Looping in play(); turn {}; {} configs at present".format(self.turn, len(self.configs)))
        for config in self.configs:
            if config.win():
                # statistic tracking
                self.leaf_count += 1
                self.player_wins[(self.turn - 1) % self.q] += 1
                # actual functionality
                winner = self.players[(self.turn - 1) % self.q]
                losers = self.players[:]
                losers.remove(winner)
                if winner:
                    winner.reward(config)
                perfect_loser = True  # flag to see if the victory was against perfect play
                for loser in losers:
                    if loser:
                        perfect_loser = False  # if any losing player was deterministic, we have proven nothing
                        loser.punish(config)
                if perfect_loser and winner:  # if a single winning strategy exists against perfect play
                    pass  # placeholder; need significant reward for Player
            elif self.turn == self.p:  # board is full; number of turns elapsed == total number of positions
                self.leaf_count += 1
                self.draw_count += 1
                for player in self.players:
                    perfect_draw = -1
                    if player:
                        perfect_draw += 1
                        player.draw(config)
                    if perfect_draw:
                        pass  # placeholder; reward Player
            else:
                player = self.players[self.turn % self.q]
                if player:
                    player_move = player.move(config.O, config.X, config.E, self.turn)
                    resulting_config = config.move(player_move)
                    self.successors.update({resulting_config})
                else:  # branch to all possible moves
                    self.successors.update(config.successors())  # add all of the configuration's successors to the set
        # omitted: identify and eliminate isomorphic boards among successors
        self.configs = self.successors
        self.successors = set()
        self.turn += 1
        if self.configs:  # if there remain any configurations to expand, then play another turn
            self.play()
        else:
            print("Exhausted all configurations.")
            print("Leaf count: {}".format(self.leaf_count))
            print("Draw count: {}".format(self.draw_count))
            for i in range(self.q):
                player = self.players[i]
                if player:
                    print("Wins for player {} ({}): {}".format(i, player.name, self.player_wins[i]))
                    print("Losses for player {} ({}): {}".format(i, player.name,
                                                                 self.leaf_count - self.draw_count - self.player_wins[
                                                                     i]))
                else:
                    print("Wins for player {} (None): {}".format(i, self.player_wins[i]))
                    print("Losses for player {} (None): {}".format(i,
                                                                   self.leaf_count - self.draw_count - self.player_wins[
                                                                       i]))
            # also include print statement to specify whether category membership for values k, n, q has been proven

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
        return moves
