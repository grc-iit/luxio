import sys,os
import pickle as pkl
from abc import ABC, abstractmethod
import pandas as pd

import pprint, warnings
pp = pprint.PrettyPrinter(depth=6)

class BehaviorClassifier(ABC):
    def __init__(self, feature_importances:pd.DataFrame, feature_categories:pd.DataFrame):
        """
        feature_importances: A dataframe where rows are features and columns are timing types (MD, READ, WRITE).
        feature_categories: A dataframe where rows are features and columns indicate how the features are to be aggregated
        """
        return

    @abstractmethod
    def fit(self, X):
        return self

    @abstractmethod
    def get_magnitude(self, X):
        return self

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


    def save(self, path):
        pkl.dump(self, open( path, "wb" ))

    @staticmethod
    def load(path):
        return pkl.load(open( path, "rb" ))
