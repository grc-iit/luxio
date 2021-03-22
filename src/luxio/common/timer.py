from time import process_time_ns
import pandas as pd

class Timer:
    def __init__(self):
        self.count = 0
        self.count_log = {}
        return

    def resume(self):
        self.start = process_time_ns()

    def pause(self):
        self.end = process_time_ns()
        self.count += self.end - self.start
        return self

    def reset(self):
        self.count = 0

    def msec(self):
        return self.count/10**6

    def log(self, id:str):
        self.count_log[id] = self.msec()
        self.reset()
        return self

    def save(self, path):
        if path is not None:
            pd.Series(self.count_log).to_csv(path)
