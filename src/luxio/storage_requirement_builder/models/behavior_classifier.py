import sys,os
import pickle as pkl
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

import pprint, warnings
pp = pprint.PrettyPrinter(depth=6)

class BehaviorClassifier(ABC):
    def __init__(self, feature_importances:pd.DataFrame):
        """
        feature_importances: A dataframe where rows are features and columns are timing types (MD, READ, WRITE).
        feature_categories: A dataframe where rows are features and columns indicate how the features are to be aggregated
        """
        feature_importances.index.name = "features"
        self.feature_importances = feature_importances.fillna(0)
        self.features = list(feature_importances.index)
        self.scores = None
        self.std_scores = None
        return

    def _smash(self, df, cols, sample_size=None):
        grp = df.groupby(cols)
        means = grp.mean().reset_index()
        stds = grp.std().reset_index().rename({orig_col:f"std_{orig_col}" for orig_col in means.columns})
        ns = grp.size().reset_index(name="count")["count"].to_numpy()/len(df)
        idxs = np.argsort(-ns)
        means = means.iloc[idxs,:]
        stds = stds.iloc[idxs,:]
        ns = ns[idxs]
        self.std_scores = [f"std_{score}" for score in self.scores]
        return means

    def _create_groups(self, df, labels, other=None, sample_size=None):
        df = pd.DataFrame(df)
        df.loc[:,"labels"] = labels
        if other is None:
            other = []
        return self._smash(df, ["labels"] + other, sample_size=sample_size)

    @abstractmethod
    def fit(self, X):
        return self

    @abstractmethod
    def get_magnitude(self, X):
        return self

    @abstractmethod
    def standardize(self, features:pd.DataFrame):
        raise Error(ErrorCode.NOT_IMPLEMENTED)

    @staticmethod
    def _use(io_identifier:pd.DataFrame, col:str):
        if col not in io_identifier:
            return 0
        return io_identifier[col]

    def save(self, path):
        pkl.dump(self, open( path, "wb" ))

    @staticmethod
    def load(path):
        return pkl.load(open( path, "rb" ))
