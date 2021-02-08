import sys,os
import pickle as pkl
import pandas as pd
import numpy as np
from .behavior_classifier import BehaviorClassifier

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
        """
        Get the extent to which an io_identifier is covered by each QoSA

        io_identifier: Either the centroid of an app class or the signature of a unique application (already standardized)
        std: The standard deviation of data around the centroid
        """
        if std is None:
            std = io_identifier/4
        #Get the distance between io_identifier and every qosa (in units of standard deviations)
        std_distance = np.absolute(self.qosas[self.scores] - io_identifier[self.scores]) / np.array(self.qosas[self.std_scores])
        #Get the extent to which this QoSA covers this io_identifier
        prob = (2 - std_distance)/2
        #Get the magnitude of the fitnesses
        prob.loc[:,"magnitude"] = self.get_magnitude(prob)
        return prob
