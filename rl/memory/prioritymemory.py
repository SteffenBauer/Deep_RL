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

    def remember(self, model, gamma, S, a, r, Snext, game_over, update):
        if self.shape is None:
            self.shape = S.shape
        entry = (S.tobytes(),a,r,Snext.tobytes(),game_over)
        if (not update) or (entry not in self.memory):
            self.memory[entry] = self.max_prio
        else:
            pred = model(np.stack((S,Snext)), training=False)
            vnext, v = pred[0][a], np.argmax(pred[1])
            new_prio = abs(r + gamma*(v) - vnext) + eps
            self.memory[entry] = new_prio
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

    def update(self, model, gamma, batch):
        S,a,r,Sn,go = zip(*batch)
        preds = model(np.stack(S+Sn), training=False)
        half = int(len(preds)/2)
        for i in range(len(batch)):
            vnext, v = preds[i][a[i]], np.argmax(preds[i+half])
            new_prio = abs(r[i] + gamma*(v) - vnext) + eps
            entry = (S[i].tobytes(),a[i],r[i],Sn[i].tobytes(),go[i])
            self.memory[entry] = new_prio

    def reset(self):
        self.memory = collections.OrderedDict()

