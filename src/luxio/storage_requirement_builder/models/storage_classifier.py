import sys,os
import pickle as pkl
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from .behavior_classifier import BehaviorClassifier
from sklearn.preprocessing import MinMaxScaler

import pprint, warnings
pp = pprint.PrettyPrinter(depth=6)
pd.options.mode.chained_assignment = None

class StorageClassifier(BehaviorClassifier):
    def __init__(self, feature_importances:pd.DataFrame, mandatory_features:List[str]=None):
        super().__init__(feature_importances, mandatory_features)
        self.qosas = None #A dataframe containing: means, stds, ns

    def fit(self, X:pd.DataFrame=None):
        self.features = ["Read_Large_BW", "Write_Large_BW", "Read_Small_BW", "Write_Small_BW", "Price"]
        self.qosas = X[self.features]
        self.transform_ = MinMaxScaler().fit(self.qosas)
        self.qosas = self.standardize(self.qosas)
        return self

    def standardize(self, qosas:pd.DataFrame):
        scaled_features = pd.DataFrame(self.transform_.transform(qosas[self.features]), columns=self.features)
        if self.scores is None:
            self.scores = []
            self.score_weights = []
            self.scores.append(f"MD_SCORE")
            self.scores.append(f"BYTES_READ")
            self.scores.append(f"BYTES_WRITTEN")
            self.score_weights.append(.33)
            self.score_weights.append(.33)
            self.score_weights.append(.33)
            self.score_weights = np.array(self.score_weights)
        qosas.loc[:,f"MD_SCORE"] = ((scaled_features["Read_Small_BW"] + scaled_features["Write_Small_BW"])/2).to_numpy()
        qosas.loc[:,f"BYTES_READ"] = scaled_features["Read_Large_BW"].to_numpy()
        qosas.loc[:,f"BYTES_WRITTEN"] = scaled_features["Write_Large_BW"].to_numpy()
        return qosas

    def get_magnitude(self, coverage:pd.DataFrame):
        return ((coverage[self.scores]*self.score_weights).sum(axis=1)/np.sum(self.score_weights)).to_numpy()

    def get_coverages(self, io_identifier:pd.DataFrame, qosas:pd.DataFrame=None) -> pd.DataFrame:
        """
        Get the extent to which an io_identifier is covered by each QoSA

        io_identifier: Either the centroid of an app class or the signature of a unique application (already standardized)
        std: The standard deviation of data around the centroid
        """
        if qosas is None:
            qosas = self.qosas
        #Get the distance between io_identifier and every qosa (in units of standard deviations)
        std_distance = 1 - np.absolute(qosas[self.scores] - io_identifier[self.scores].to_numpy())
        #Get the magnitude of the fitnesses
        std_distance.loc[:,"magnitude"] = self.get_magnitude(std_distance)
        #Add features
        std_distance.loc[:,self.features] = qosas[self.features].to_numpy()
        return std_distance
