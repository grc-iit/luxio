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
        X = X.iloc[0:100,:]
        #Identify clusters of transformed data
        self.transform_ = ChainTransformer([LogTransformer(base=10,add=1), MinMaxScaler()])
        X_features = self.transform_.fit_transform(X[self.features])*self.feature_importances.max(axis=1).to_numpy()
        #for k in [4, 6, 8, 10, 15, 30, 50]:
        #    self.model_ = KMeans(k=k)
        #    self.labels_ = self.model_.fit_predict(X_features)
        #    print(f"SCORE k={k}: {self.score(X_features, self.labels_)}")
        self.model_ = KMeans(k=10)
        self.labels_ = self.model_.fit_predict(X_features)
        #Cluster non-transformed data
        self.app_classes = self.standardize(X)
        self.app_classes = self._create_groups(self.app_classes, self.labels_, other=self.mandatory_features)
        print(self.app_classes)
        return self

    def standardize(self, qosas:pd.DataFrame):
        scaled_features = pd.DataFrame(self.transform_.transform(qosas[self.features].astype(float)), columns=self.features)
        io_identifier.loc[:,"MD_SCORE"] = (scaled_features[MD]).sum(axis=1).to_numpy()
        io_identifier.loc[:,"READ_OPS"] = (scaled_features[READ_OPS]).sum(axis=1).to_numpy()
        io_identifier.loc[:,"BYTES_READ"] = (scaled_features[BYTES_READ]).sum(axis=1).to_numpy()
        io_identifier.loc[:,"BYTES_WRITTEN"] = (scaled_features[BYTES_WRITTEN]).sum(axis=1).to_numpy()
        io_identifier.loc[:,"SCALE"] = (scaled_features[SCALE]).sum(axis=1).to_numpy()

        if self.scores is None:
            self.scores = ["MD_SCORE", "READ_OPS", "BYTES_READ", "BYTES_WRITTEN", "SCALE"]
            self.score_weights = np.array([1]*len(self.scores))
        return io_identifier

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
