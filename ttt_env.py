import time

import gym.core as gc
from board_naive import Board


class nkq_game(gc.Env):
    def __init__(self, n=3, k=2):
        self.n = n
        self.k = k
        self.p = self.n ** self.k
        self.board = None

    def step(self, action, display_board=False):
        # translate action to a move
        move = []
        info = {}
        action_ = action
        while action > 0:
            quo = action // self.n
            rem = action % self.n
            move.append(rem)
            action = quo
            # then, append zeroes to grow to size
        while len(move) < self.k:
            move.append(0)
        move = list(reversed(move)) # reverse; indices were at fault previously
        illegal_move = not self.board.move_available(move)
        if illegal_move:
            print("\nIllegal move {} = {} attempted on:".format(action_, move))
            self.board.print_2d()
            time.sleep(0.03)
            raise ValueError("you weren't supposed to do that")
        self.board.move(move)
        if display_board:
            print("Board after move {} by brain:".format(move))
            self.board.print_2d()
        # if game is over, return here
        if self.board.draw() or self.board.win():
            reward = 20   # against random play, penalize draws
            if self.board.win():
                reward = 120  # don't overincentivize risky play - win > draw by less than loss > win
            observation = self.board.to_linear_array()
            done = True
            return observation, reward, done, info
        else:
            self.board.rand_move()
            if display_board:
                print("Board after random move:")
                self.board.print_2d()
            observation = self.board.to_linear_array()
            if self.board.draw():
                done = True
                reward = 20    # against random play, penalize draws
                return observation, reward, done, info
            if self.board.win():
                reward = -100  # punish losses heavily on this board - draw is *always* possible
                done = True
                return observation, reward, done, info
            done = False
            reward = -1
            return observation, reward, done, info

    def reset(self):
        self.board = Board.blank_board(n=self.n, k=self.k)
        return self.board.to_linear_array()

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        pass

