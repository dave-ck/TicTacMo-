import time

import gym.core as gc
from env_config import Config


class nkq_game(gc.Env):
    def __init__(self, n=3, k=2):
        self.n = n
        self.k = k
        self.p = self.n ** self.k
        self.config = None

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
        illegal_move = not self.config.move_available(move)
        if illegal_move:
            print("\nIllegal move {} = {} attempted on:".format(action_, move))
            self.config.print_2d()
            time.sleep(0.03)
            raise ValueError("you weren't supposed to do that")
        self.config.move(move)
        if display_board:
            print("Board after move {} by brain:".format(move))
            self.config.print_2d()
        # if game is over, return here
        if self.config.draw() or self.config.win():
            reward = -10   # against random play, penalize draws
            if self.config.win():
                reward = 120  # don't overincentivize risky play - win > draw by less than loss > win
            observation = self.config.to_linear_array()
            done = True
            return observation, reward, done, info
        else:
            self.config.rand_move()
            if display_board:
                print("Board after random move:")
                self.config.print_2d()
            observation = self.config.to_linear_array()
            if self.config.draw():
                done = True
                reward = -10    # against random play, penalize draws
                return observation, reward, done, info
            if self.config.win():
                reward = -100  # punish losses heavily on this board - draw is *always* possible
                done = True
                return observation, reward, done, info
            done = False
            reward = -1
            return observation, reward, done, info

    def reset(self):
        self.config = Config(n=self.n, k=self.k, p=self.p, O=[], X=[], E=self.enumerate_moves(), turn=0)
        return self.config.to_linear_array()

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        pass

    def enumerate_moves(self):  # Cimpl; todo: move to Config class
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
