from .memory import Memory

import random
import numpy as np

class PrioMemory(Memory):
    def __init__(self, model, memory_size=65536, zeta=0.6, beta=0.4):
        self.model = model
        self.memory_size = memory_size
        self.zeta = zeta
        self.beta = beta
        self.memory = list()
        self.hashes = set()
        self.max_prio = 1.0
        pass

    def remember(self, S, a, r, Snext, game_over):
        h = hash((S.tostring(), a, r, Snext.tostring(), game_over))
        if h not in self.hashes:
            self.memory.append([(S,a,r,Snext,game_over), self.max_prio])
            self.hashes.add(h)
        else:
            pass
        if self.memory_size is not None and len(self.memory) > self.memory_size:
            Sr, Ar, Rr, SNr, GOr = self.memory.pop(0)[0]
            h = hash((Sr.tostring(), Ar, Rr, SNr.tostring(), GOr))
            self.hashes.remove(h)

        pass

    def get_batch(self, batch_size):
        pass

    def update(self, batch):
        pass

    def reset(self):
        pass

