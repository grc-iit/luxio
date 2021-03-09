from time import process_time_ns
import pandas as pd

class Timer:
    log = {}

    def __init__(self):
        self.count = 0
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
        self.log[id] = self.msec()
        self.reset()

    def save(self, path):
        pd.DataFrame(self.log).to_csv(path)
