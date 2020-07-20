from .callbacks import Callback

class Checkpoint(Callback):
    def __init__(self, interval=1):
        self.interval = interval
    def epoch_end(self, stats):
        if stats['epoch'] % self.interval == 0:
            filename = '{}_{:03d}.h5'.format(stats['name'], stats['epoch'])
            stats['model'].save(filename)

