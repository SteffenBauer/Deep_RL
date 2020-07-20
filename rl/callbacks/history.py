from .callbacks import Callback

import time

class HistoryLog(Callback):
    def __init__(self, name, params=None):
        st = time.gmtime()
        self.timestamp = "{:04d}{:02d}{:02d}_{:02d}{:02d}{:02d}".format(st.tm_year, st.tm_mon, st.tm_mday, st.tm_hour, st.tm_min, st.tm_sec)
        self.filename = '{}-{}.log'.format(name, self.timestamp)
        with open(self.filename, 'w+') as fp:
            if params is not None:
                for k,v in sorted(params.items()):
                    fp.write("{:18s}: {}\n".format(k,v))
                fp.write("\n")
            fp.write('Epoch, Epsilon, Win Ratio, Avg Score, Max Score, Avg Turns, Max Turns,  Memory,  Time,     Timestamp\n')
    def epoch_end(self, stats):
        st = time.gmtime()
        stats['timestamp'] = "{:04d}-{:02d}-{:02d}_{:02d}-{:02d}-{:02d}".format(st.tm_year, st.tm_mon, st.tm_mday, st.tm_hour, st.tm_min, st.tm_sec)
        with open(self.filename, 'a') as fp:
            fp.write('{epoch:> 5d}, {epsilon:>7.2f}, {win_ratio:>9.2%}, {avg_score:>9.2f}, {max_score:>9.2f}, {avg_turns:>9.2f}, {max_turns:>9.2f}, {memory_fill:>8d}, {epoch_time:>5.0f}, {timestamp}\n'.format(**stats))

