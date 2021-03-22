import sys,os
import pickle as pkl
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
from luxio.common.configuration_manager import *

import pprint, warnings
pp = pprint.PrettyPrinter(depth=6)

class BehaviorClassifier(ABC):
    def __init__(self, features:list, score_conf:dict, feature_importances:pd.DataFrame, mandatory_features:List[str]=None):
        """
        feature_importances: A dataframe where rows are features and columns are timing types (MD, READ, WRITE).
        feature_categories: A dataframe where rows are features and columns indicate how the features are to be aggregated
        """

        self.features = list(features)
        if feature_importances is not None:
            self.feature_importances = feature_importances.fillna(0).transpose()
            self.features = list(self.feature_importances.columns)
        else:
            self.feature_importances = pd.DataFrame([[1]*len(self.features)], columns=self.features, index=[0])
        self.feature_importances /= float(self.feature_importances.sum(axis=1))
        self.score_conf = score_conf
        self.mandatory_features = mandatory_features if mandatory_features is not None else []
        self.scores = None
        self.conf = ConfigurationManager.get_instance()

    def _smash(self, df:pd.DataFrame, cols:np.array):
        grp = df.groupby(cols)
        medians = grp.median().reset_index()
        #std_col_map = {orig_col:f"std_{orig_col}" for orig_col in means.columns}
        #std_cols = list(std_col_map.values())
        #stds = grp.std().reset_index().rename(std_col_map)
        ns = grp.size().reset_index(name="count")["count"].to_numpy()/len(df)
        idxs = np.argsort(-ns)
        medians = medians.iloc[idxs,:]
        #stds = stds.iloc[idxs,:]
        ns = ns[idxs]
        #means.loc[:,std_cols] = stds.to_numpy()
        medians.loc[:,"count"] = ns
        return medians

    def _create_groups(self, df:pd.DataFrame, labels:np.array, other:List[str]=None):
        df = pd.DataFrame(df)
        df.loc[:,"labels"] = labels
        if other is None:
            other = []
        return self._smash(df, ["labels"] + other)

    @abstractmethod
    def fit(self, X):
        return self

    @abstractmethod
    def get_magnitude(self, X):
        return self

    @abstractmethod
    def standardize(self, features:pd.DataFrame):
        raise Error(ErrorCode.NOT_IMPLEMENTED)

    def save(self, path):
        pkl.dump(self, open( path, "wb" ))

    @staticmethod
    def load(path):
        return pkl.load(open( path, "rb" ))
