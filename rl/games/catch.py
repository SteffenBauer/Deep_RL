import random
import numpy as np
from .game import Game

class Catch(Game):

    def __init__(self, grid_size=10, with_penalty=False, split_reward=False, hop=0.0):
        self.grid_size = grid_size
        self.with_penalty = with_penalty
        self.split_reward = split_reward
        self.hop = hop
        self.reset()

    def reset(self):
        fx = random.randrange(0, self.grid_size-1)
        bx = random.randrange(1, self.grid_size-2)
        self.state = [0, fx, bx]
        self.penalty = 0.0
        return self.get_frame()

    @property
    def name(self):       return "Catch"
    @property
    def nb_actions(self): return 3
    @property
    def actions(self):    return {0: 'left', 1: 'skip', 2: 'right'}

    def play(self, action):
        if self.is_over() or action not in range(self.nb_actions):
            return (self.get_frame(), self.get_score(), self.is_over())
        fy, fx, basket = self.state
        if action == 0:
            new_basket = max(1, basket-1)
        elif action == 2:
            new_basket = min(self.grid_size-2, basket+1)
        else:
            new_basket = basket
        if random.random() < self.hop:
            if random.random() <= 0.5:
                fx = max(0, fx-1)
            else:
                fx = min(self.grid_size-1, fx+1)
        self.state = [fy+1, fx, new_basket]
        self.penalty = 0.0 if action == 1 else -0.05
        return (self.get_frame(), self.get_score(), self.is_over())

    def get_state(self):
        fy, fx, basket = self.state
        canvas = np.zeros((self.grid_size,self.grid_size,3))
        canvas[-1, basket-1:basket + 2, :] = (1,1,0)
        if self.is_won():
            canvas[fy, fx, :] = (0,1,0)
        elif self.is_over():
            canvas[fy, fx, :] = (1,0,0)
        else:
            canvas[fy, fx, :] = (1,1,0)

        return canvas

    def get_score(self):
        if self.is_won():
            return self.win_reward()
        elif self.is_over():
            return -1.0
        else:
            return self.penalty if self.with_penalty else 0.0

    def win_reward(self):
        if not self.split_reward:
            return 1.0
        fy, fx, basket = self.state
        return 1.0 if (fx == basket) else 0.5
        
    def is_over(self):
        return self.state[0] == self.grid_size-1

    def is_won(self):
        fy, fx, basket = self.state
        return self.is_over() and abs(fx - basket) <= 1

