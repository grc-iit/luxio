import sys,os
import pickle as pkl
import pandas as pd
import numpy as np
from .behavior_classifier import BehaviorClassifier
<<<<<<< HEAD

import pprint, warnings
pp = pprint.PrettyPrinter(depth=6)

class StorageClassifier(BehaviorClassifier):
    def __init__(self, feature_importances:pd.DataFrame, feature_categories:pd.DataFrame):

        #TODO: Remove example
        self.features = ["c", "d"]
        self.feature_importances = pd.DataFrame({
            "MD_TIME" : {"c" : .25, "d" : .25},
            "READ_TIME" : {"c" : .75, "d" : .25},
            "WRITE_TIME" : {"c" : .25, "d" : .75}
        })
        self.feature_categories = pd.DataFrame({
            "c" : {"mandatory" : True, "category" : 0},
            "d" : {"mandatory" : False, "category" : 1}
        })

        super().__init__(feature_importances, feature_categories)
        self.qosas = None #A dataframe containing: means, stds, ns

    def fit(self, X:pd.DataFrame):
        self.qosas = pd.DataFrame([{"c": 25*i, "d": 5, "std_c": 5, "std_d": 2, "n": 100} for i in range(10)])
        self.stds = ["std_c", "std_d"]
        self.qosas = self.standardize(self.qosas)
        return self

    def get_magnitude(self, coverage:pd.DataFrame):
        return 1

    def get_coverages(self, io_identifier:pd.DataFrame, std:pd.DataFrame=None) -> pd.DataFrame:
=======
from sklearn.preprocessing import MinMaxScaler

import pprint, warnings
pp = pprint.PrettyPrinter(depth=6)
pd.options.mode.chained_assignment = None

class StorageClassifier(BehaviorClassifier):
    def __init__(self, feature_importances:pd.DataFrame):
        super().__init__(feature_importances)
        self.qosas = None #A dataframe containing: means, stds, ns

    def fit(self, X:pd.DataFrame=None):
        self.features = ["Read_Large_BW", "Write_Large_BW", "Read_Small_BW", "Write_Small_BW"]
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
>>>>>>> 3a568e3eddc60589a3c644a83c2fab657537eca8
        """
        Get the extent to which an io_identifier is covered by each QoSA

        io_identifier: Either the centroid of an app class or the signature of a unique application (already standardized)
        std: The standard deviation of data around the centroid
        """
<<<<<<< HEAD
        if std is None:
            std = io_identifier/4
        #Get the distance between io_identifier and every qosa (in units of standard deviations)
        std_distance = np.absolute(self.qosas[self.scores] - io_identifier[self.scores]) / np.array(self.qosas[self.std_scores])
        #Get the extent to which this QoSA covers this io_identifier
        prob = (2 - std_distance)/2
        #Get the magnitude of the fitnesses
        prob.loc[:,"magnitude"] = self.get_magnitude(prob)
        return prob
=======
        if qosas is None:
            qosas = self.qosas
        #Get the distance between io_identifier and every qosa (in units of standard deviations)
        std_distance = 1 - np.absolute(qosas[self.scores] - io_identifier[self.scores].to_numpy())
        #Get the magnitude of the fitnesses
        std_distance.loc[:,"magnitude"] = self.get_magnitude(std_distance)
        #Add features
        std_distance.loc[:,self.features] = qosas[self.features].to_numpy()
        return std_distance
>>>>>>> 3a568e3eddc60589a3c644a83c2fab657537eca8
