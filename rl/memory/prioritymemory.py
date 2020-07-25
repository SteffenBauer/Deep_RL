from .memory import Memory

import collections
import random
import numpy as np

eps = 1e-10

class PrioMemory(Memory):
    def __init__(self, model, memory_size=65536):
        self.model = model
        self.memory_size = memory_size
        self.memory = collections.OrderedDict()
        self.max_prio = 1.0
        self.shape = None
        pass

    def remember(self, S, a, r, Snext, game_over):
        if self.shape is None:
            self.shape = S.shape
        entry = (S.tobytes(),a,r,Snext.tobytes(),game_over)
        if entry not in self.memory:
            self.memory[entry] = self.max_prio
        if self.memory_size is not None and len(self.memory) > self.memory_size:
            self.memory.popitem(last=False)

    def get_batch(self, batch_size, zeta=0.6, beta=0.4):
        entries = list(self.memory.keys())
        probs = list(self.memory.values())

        prob = np.array(probs) ** zeta
        prob = prob / np.sum(prob)

        chosen_idx = np.random.choice(len(self.memory), p=prob, size=batch_size, replace=False)
        chosen_entries = [entries[idx] for idx in chosen_idx]
        batch = [(np.frombuffer(S).reshape(self.shape),a,r,np.frombuffer(Sn).reshape(self.shape),game_over) for S,a,r,Sn,game_over in chosen_entries]

        n = len(self.memory)
        weights = np.array([(n*probs[idx]) ** (-beta) for idx in chosen_idx])
        return batch, weights

    def update(self, gamma, batch):
        S,a,r,Sn,go = zip(*batch)
        preds = self.model.predict(np.array(S+Sn))
        half = int(len(preds)/2)
        for i in range(len(batch)):
            v, vn = preds[i][a[i]], np.argmax(preds[i+half])
            if go[i]:
                new_prio = abs(r[i] - v) + eps
            else:
                new_prio = abs(r[i] + gamma*(vn) - v) + eps
            entry = (S[i].tobytes(),a[i],r[i],Sn[i].tobytes(),go[i])
            self.memory[entry] = new_prio

    def reset(self):
        self.memory = collections.OrderedDict()

