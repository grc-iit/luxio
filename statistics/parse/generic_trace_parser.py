
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from functools import reduce

class GenericTraceParser(ABC):
    def __init__(self):
        return

    def preprocess(self, out=None):
        self.standardize()
        self.clean()
        self.to_csv(out)

    def _project(self, map_path):
        map = self.get_mappings(map_path)
        self.df = self.df.rename(columns=map)

    @abstractmethod
    def standardize(self):
        raise Exception("preprocess not implemented")

    @abstractmethod
    def clean(self):
        raise Exception("cleaning unimplemented")

    @staticmethod
    def combine(dfs):
        if len(dfs) == 1:
            return dfs[0]
        common = reduce(lambda df1,df2 : np.intersect1d(df1.columns, df2.columns), dfs)
        dfs = [df[common] for df in dfs]
        df = pd.concat(dfs)
        return df

    @staticmethod
    def get_mappings(path):
        mappings = pd.read_csv(path)
        map = {orig_name : new_name for orig_name, new_name in zip(list(mappings["original"]), list(mappings["new"]))}
        return map

    def to_csv(self, path):
        if path == None:
            return
        self.df.to_csv(path, index=False)
