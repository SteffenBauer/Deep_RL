from .memory import Memory

import collections
import random
import numpy as np

eps = 1e-20

class PrioMemory(Memory):
    def __init__(self, model, memory_size=65536, zeta=0.6, beta=0.4):
        self.model = model
        self.memory_size = memory_size
        self.zeta = zeta
        self.beta = beta
        self.memory = collections.OrderedDict()
        self.max_prio = 1.0
        pass

    def remember(self, model, gamma, S, a, r, Snext, game_over, update=True):
        entry = (S,a,r,Snext,game_over)
        if (not update) or (h not in self.memory):
            self.memory[entry] =  self.max_prio
        else:
            pred = model([S, Snext])
            new_prio = abs(r + gamma*(v) - vnext) + eps
            self.memory[entry] = new_prio
        if self.memory_size is not None and len(self.memory) > self.memory_size:
            self.memory.popitem(last=False)

        pass

    def get_batch(self, batch_size):
        numpy.random.choice(list(d.keys()), p=prob, size=2, replace=False)
        pass

    def update(self, batch):
        pass

    def reset(self):
        pass

