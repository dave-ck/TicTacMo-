from obsolete.board_naive import Board
import time


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
        self.boards = {Board.blank_board(self.n, self.k)}
        self.successors = set()
        self.turn = 0
        self.leaf_count = 0  # keep track of total games ended
        self.draw_count = 0  # keep track of total draws to save computing through summation/subtraction at end
        self.player_wins = {i: 0 for i in range(q)}  # keep track of how many wins each player (play position) has
        self.initTime = time.time()
        print("\n\nBegan processing board with n={}, k={}, q={}".format(self.n, self.k, self.q))

    def play(self):
        # for every config
        print("Looping in play(); {}s since init; turn {}; {} configs at present".format(time.time() - self.initTime,
                                                                                         self.turn, len(self.boards)))
        for board in self.boards:
            # board.print_2d()
            if board.win():
                # statistic tracking
                self.leaf_count += 1
                self.player_wins[(self.turn - 1) % self.q] += 1
                # actual functionality
                winner = self.players[(self.turn - 1) % self.q]
                losers = self.players[:]
                losers.remove(winner)
                if winner:
                    winner.reward(board)
                perfect_loser = True  # flag to see if the victory was against perfect play
                for loser in losers:
                    if loser:
                        perfect_loser = False  # if any losing player was deterministic, we have proven nothing
                        loser.punish(board)
                if perfect_loser and winner:  # if a single winning strategy exists against perfect play
                    pass  # placeholder; need significant reward for Player
            elif self.turn == self.p:  # board is full; number of turns elapsed == total number of positions
                self.leaf_count += 1
                self.draw_count += 1
                for player in self.players:
                    perfect_draw = -1
                    if player:
                        perfect_draw += 1
                        player.draw(board)
                    if perfect_draw:
                        pass  # placeholder; reward Player
            else:
                player = self.players[self.turn % self.q]
                if player:
                    player_move = player.move(board.O, board.X, board.E, self.turn)
                    resulting_config = board.move_clone(player_move)
                    self.successors.update({resulting_config})
                else:  # branch to all possible moves
                    self.successors.update(board.successors())  # add all of the configuration's successors to the set
        # omitted: identify and eliminate isomorphic boards among successors
        self.boards = self.successors
        self.successors = set()
        self.turn += 1
        if self.boards:  # if there remain any configurations to expand, then play another turn
            self.play()
        else:
            print("Exhausted all configurations; {}s since init.".format(time.time()-self.initTime))
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
