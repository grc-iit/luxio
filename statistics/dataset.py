
import pandas as pd

class Dataset:
    def __init__(self):
        self.df = None
        self.train_x = None
        self.train_y = None
        self.hyper_x = None
        self.hyper_y = None
        self.test_x = None
        self.test_y = None

    def read_csv(path):
        self.df = pd.read_csv(path)
