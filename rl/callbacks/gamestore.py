from .callbacks import Callback
import collections

class GameStore(Callback):
    def __init__(self):
        self.gameQueue = collections.deque([], 1)
        self.gameQueue.clear()
    def game_start(self, frame):
        self.episode = [(frame, None, 0.0, False)]
    def game_step(self, frame, action, reward, isOver):
        self.episode.append((frame, action, reward, isOver))
    def game_over(self):
        self.gameQueue.clear()
        self.gameQueue.append(self.episode)

