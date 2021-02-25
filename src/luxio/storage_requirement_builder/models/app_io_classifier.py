import sys,os
from .behavior_classifier import BehaviorClassifier
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from clever.transformers import *
from clever.models.cluster import *

import pprint, warnings
pp = pprint.PrettyPrinter(depth=6)
pd.options.mode.chained_assignment = None

class AppClassifier(BehaviorClassifier):
    def __init__(self, feature_importances:pd.DataFrame, mandatory_features:List[str]=None):
        super().__init__(feature_importances, mandatory_features)
        self.app_classes = None #A pandas dataframe containing: means, stds, number of entries, and qosas
        self.thresh = .25

    def fit(self, X:pd.DataFrame):
        """
        Identify groups of application behavior from a dataset of traces using the features.
        Calculate a standardized set of scores that are common between the apps and QoSAs
        """
        self.app_classes = pd.DataFrame([{"a": 25*i, "b": 5, "std_a": 5, "std_b": 2, "n": 100} for i in range(10)])
        self.stds = ["std_a", "std_b"]
        self.app_classes = self.standardize(self.app_classes)
        return self

    def filter_qosas(self, storage_classifier):
        """
        For each application class, filter out QoSAs that have little chance of being useful.
        """
        qosas = []
        #Compare every application class with every qosa
        for idx,app_class in self.app_classes.iterrows():
            coverages = storage_classifier.get_coverages(app_class[self.scores], std=app_class[self.std_scores])
            coverages = coverages[coverages.magnitude > self.thresh]
            qosas.append(coverages)
        #Add the qosas to the dataframe
        self.app_classes.loc[:,"qosas"] = pd.Series(qosas)

    def get_magnitude(self, fitness:pd.DataFrame):
        """
        Convert the fitness vector into a single score.

        Some features are mandatory, and will cause fitness to be 0 if not met.
        Some features are continuous, and have a spectrum of values
        """
        return 1

    def get_fitnesses(self, io_identifier:pd.DataFrame) -> pd.DataFrame:
        """
        Determine how well the I/O Identifier fits within each class of behavior
        """
        #Calculate the scores
        io_identifier = self.standardize(io_identifier)
        #Get the distance between io_identifier and every app class (in units of standard deviations)
        std_distance = np.absolute(self.app_classes[self.scores] - io_identifier[self.scores]) / np.array(self.app_classes[self.std_scores])
        #Get probability of getting this io_identifier from the set of qosas (TODO: calculate p val from std_distance)
        prob = (2 - std_distance)/2
        #Get the magnitude of the fitnesses
        prob.loc[:,"magnitude"] = self.get_magnitude(prob)
        #Add qosas to dataframe
        prob.loc[:,"qosas"] = self.app_classes["qosas"]
        return prob
