import time
import gym.core as gc
from board import Board


class nkq_game(gc.Env):
    def __init__(self, n=3, k=2):
        self.n = n
        self.k = k
        self.p = self.n ** self.k
        self.board = None

    def step(self, action, display_board=False):
        # translate action to a move
        info = {}
        illegal_move = not self.board.move_available(action)
        if illegal_move:
            print("\nIllegal move {} = {} attempted on:".format(action, move))
            self.board.print_2d()
            time.sleep(0.03)
            raise ValueError("you weren't supposed to do that")
        self.board.move(action)
        if display_board:
            print("Board after move {} by brain:".format(action))
            self.board.print_2d()
        # if game is over, return here
        if self.board.draw():
            reward = 1   # against random play, penalize draws
            observation = self.board.to_linear_array()
            done = True
            return observation, reward, done, info
        elif self.board.win(1) or self.board.win(-1):
            reward = 5  # don't overincentivize risky play - win > draw by less than loss > win
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
                reward = 1    # against random play, penalize draws
                return observation, reward, done, info
            if self.board.win(1) or self.board.win(-1):
                reward = -100  # punish losses heavily on this board - draw is *always* possible
                done = True
                return observation, reward, done, info
            done = False
            reward = 0
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

