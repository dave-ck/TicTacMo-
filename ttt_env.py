import time

import gym.core as gc
from env_config import Config


class nkq_game(gc.Env):
    def __init__(self, n=3, k=2, q=2):
        self.n = n
        self.k = k
        self.q = q
        self.p = self.n ** self.k
        self.config = None

    def action_to_move(self, action):
        # translate action to a move
        move = []
        while action > 0:
            quo = action // self.n
            rem = action % self.n
            move.append(rem)
            action = quo
            # then, append zeroes to grow to size
        while len(move) < self.k:
            move.append(0)
        move = list(reversed(move))  # reverse; indices were at fault previously
        return move

    def step(self, action, display_board=False):
        move = self.action_to_move(action)
        if not self.config.move_available(move):
            print("Error board print:")
            print(action)
            self.config.print_2d()
            raise ValueError("Cannot make move {} on config X={}, O={}".format(move, self.config.X, self.config.O))
        # left off HERE
        # illegal move (last time [0,2]) being made by brain - shouldn't happen at all
        self.config.move(move)
        if display_board:
            print("Board after move {} by brain:".format(move))
            self.config.print_2d()
        if self.config.draw() or self.config.win():
            done = True
        else:
            done = False
        reward = self.config.reward(self.q)  # can pass win_forcer param here
        observation = self.config.to_linear_array()
        info = {}
        return observation, reward, done, info

    def rand_step(self, display_board=False):
        info = {}
        self.config.rand_move()
        if display_board:
            print("Board after random move:")
            self.config.print_2d()
        observation = self.config.to_linear_array()
        if self.config.draw():
            done = True
            reward = self.config.reward(self.q)
        elif self.config.win():
            done = True
            reward = -1000  # todo: use function of k and n for rewards
        else:
            done = False
            reward = -1
        observation = self.config.to_linear_array()

        return observation, reward, done, info

    def old_step(self, action, display_board=False):
        info = {}
        move = self.action_to_move(action)
        illegal_move = not self.config.move_available(move)
        if illegal_move:
            print("\nIllegal move {} = {} attempted on:".format(action, move))
            self.config.print_2d()
            time.sleep(0.03)
            raise ValueError("you weren't supposed to do that")
        self.config.move(move)
        if display_board:
            print("Board after move {} by brain:".format(move))
            self.config.print_2d()
        # if game is over, return here
        if self.config.draw() or self.config.win():
            reward = -10  # against random play, penalize draws
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
                reward = -10  # against random play, penalize draws
                return observation, reward, done, info
            if self.config.win():
                reward = -1000  # punish losses heavily on every 2-player board - draw is *always* possible
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
