import sys,os
import pickle as pkl
from abc import ABC, abstractmethod
<<<<<<< HEAD
import pandas as pd
=======
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
>>>>>>> 3a568e3eddc60589a3c644a83c2fab657537eca8

import pprint, warnings
pp = pprint.PrettyPrinter(depth=6)

class BehaviorClassifier(ABC):
<<<<<<< HEAD
    def __init__(self, feature_importances:pd.DataFrame, feature_categories:pd.DataFrame):
=======
    def __init__(self, feature_importances:pd.DataFrame):
>>>>>>> 3a568e3eddc60589a3c644a83c2fab657537eca8
        """
        feature_importances: A dataframe where rows are features and columns are timing types (MD, READ, WRITE).
        feature_categories: A dataframe where rows are features and columns indicate how the features are to be aggregated
        """
<<<<<<< HEAD
        return

=======
        feature_importances.index.name = "features"
        self.feature_importances = feature_importances.fillna(0)
        self.features = list(feature_importances.index)
        self.scores = None
        self.std_scores = None
        return

    def _smash(self, df:pd.DataFrame, cols:np.array):
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

    def _create_groups(self, df:pd.DataFrame, labels:np.array, other:List[str]=None):
        df = pd.DataFrame(df)
        df.loc[:,"labels"] = labels
        if other is None:
            other = []
        return self._smash(df, ["labels"] + other)

>>>>>>> 3a568e3eddc60589a3c644a83c2fab657537eca8
    @abstractmethod
    def fit(self, X):
        return self

    @abstractmethod
    def get_magnitude(self, X):
        return self

<<<<<<< HEAD
    def standardize(self, features:pd.DataFrame):
        """
        Calculate a standardized set of scores based on the features.
        """

        #TODO: An example, need to actually implement
        features.loc[:,"score"] = features[self.features[0]] * features[self.features[1]]
        features.loc[:, "std_score"] = 5
        self.scores = ["score"]
        self.std_scores = ["std_score"]
        return features

=======
    @abstractmethod
    def standardize(self, features:pd.DataFrame):
        raise Error(ErrorCode.NOT_IMPLEMENTED)
>>>>>>> 3a568e3eddc60589a3c644a83c2fab657537eca8

    def save(self, path):
        pkl.dump(self, open( path, "wb" ))

    @staticmethod
    def load(path):
        return pkl.load(open( path, "rb" ))
