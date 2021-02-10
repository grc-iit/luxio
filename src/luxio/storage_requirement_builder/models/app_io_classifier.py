import sys,os
from .behavior_classifier import BehaviorClassifier
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from clever.transformers import *

import pprint, warnings
pp = pprint.PrettyPrinter(depth=6)
pd.options.mode.chained_assignment = None

class AppClassifier(BehaviorClassifier):
    def __init__(self, feature_importances:pd.DataFrame):
        super().__init__(feature_importances)
        self.app_classes = None #A pandas dataframe containing: means, stds, number of entries, and qosas
        self.thresh = .25

    def fit(self, X:pd.DataFrame):
        """
        Identify groups of application behavior from a dataset of traces using the features.
        Calculate a standardized set of scores that are common between the apps and QoSAs
        """
        X = X.iloc[0:100,:]
        self.transform_ = ChainTransformer([LogTransformer(base=10,add=1), MinMaxScaler()])
        X_features = self.transform_.fit_transform(X[self.features])*self.feature_importances.max(axis=1).to_numpy()
        self.model_ = KMeans(n_clusters=32)
        self.labels_ = self.model_.fit_predict(X_features)
        self.app_classes = self.standardize(X)
        self.app_classes = self._create_groups(self.app_classes, self.labels_)
        return self

    def filter_qosas(self, storage_classifier, top_n=10):
        """
        For each application class, filter out QoSAs that have little chance of being useful.
        """
        qosas = []
        app_classes = []
        #Compare every application class with every qosa
        for idx,app_class in self.app_classes.iterrows():
            coverages = storage_classifier.get_coverages(app_class[self.scores])
            coverages = coverages.nlargest(top_n, "magnitude")
            qosas.append(coverages)
        #Add the qosas to the dataframe
        #self.app_classes.loc[:,"qosas"] = pd.Series(qosas)
        self.qosas = pd.concat(qosas)
        self.qosas.index.name=  "qosa_id"
        self.app_qosa_mapping = pd.concat([self.app_classes]*top_n)

    def standardize(self, io_identifier):
        MD = ["TOTAL_STDIO_OPENS", "TOTAL_POSIX_OPENS", "TOTAL_MPIIO_COLL_OPENS", "TOTAL_POSIX_STATS", "TOTAL_STDIO_SEEKS"]
        READ_OPS = ["TOTAL_STDIO_READS", "TOTAL_POSIX_SIZE_READ_0_100"]
        WRITE_OPS = ["TOTAL_WRITE_OPS", "TOTAL_POSIX_WRITES", "TOTAL_MPIIO_SIZE_WRITE_AGG_10K_100K", "TOTAL_MPIIO_SIZE_WRITE_AGG_0_100"]
        BYTES_READ = ["TOTAL_BYTES_READ"]
        BYTES_WRITTEN = ["TOTAL_BYTES_WRITTEN"]
        SEQUENTIAL_IO = ["TOTAL_POSIX_SEQ_WRITES"]
        SCALE = ["NPROCS"]
        scaled_features = pd.DataFrame(self.transform_.transform(io_identifier[self.features].astype(float)), columns=self.features)
        if self.scores is None:
            self.scores = []
            self.score_weights = []
            self.scores.append("MD_SCORE")
            self.scores.append("BYTES_READ")
            self.scores.append("BYTES_WRITTEN")
            self.score_weights.append(np.sum(self.feature_importances["TOTAL_MD_TIME"].transpose()[MD].to_numpy()))
            self.score_weights.append(np.sum(self.feature_importances["TOTAL_READ_TIME"].transpose()[BYTES_READ].to_numpy()))
            self.score_weights.append(np.sum(self.feature_importances["TOTAL_WRITE_TIME"].transpose()[BYTES_WRITTEN].to_numpy()))
            self.score_weights.append(np.sum(self.feature_importances["SCALE"].transpose()[SCALE].to_numpy()))
            self.score_weights = np.array(self.score_weights)
        io_identifier.assign()
        io_identifier.loc[:,"MD_SCORE"] = (scaled_features[MD] * self.feature_importances["TOTAL_MD_TIME"].transpose()[MD].to_numpy()).sum(axis=1).to_numpy()
        io_identifier.loc[:,"BYTES_READ"] = (scaled_features[BYTES_READ] * self.feature_importances["TOTAL_READ_TIME"].transpose()[BYTES_READ].to_numpy()).sum(axis=1).to_numpy()
        io_identifier.loc[:,"BYTES_WRITTEN"] = (scaled_features[BYTES_WRITTEN] * self.feature_importances["TOTAL_WRITE_TIME"].transpose()[BYTES_WRITTEN].to_numpy()).sum(axis=1).to_numpy()
        #print(io_identifier[self.scores])
        return io_identifier

    def get_magnitude(self, fitness:pd.DataFrame):
        """
        Convert the fitness vector into a single score.

        Some features are mandatory, and will cause fitness to be 0 if not met.
        Some features are continuous, and have a spectrum of values
        """
        return ((fitness[self.scores]*self.score_weights).sum(axis=1)/np.sum(self.score_weights)).to_numpy()

    def get_fitnesses(self, io_identifier:pd.DataFrame) -> pd.DataFrame:
        """
        Determine how well the I/O Identifier fits within each class of behavior
        """
        #Calculate the scores
        #print(f"\n\n\n\nH1: {io_identifier}")
        io_identifier = self.standardize(io_identifier[self.features])
        #print(f"H2: {io_identifier[self.scores]}")
        #Get the distance between io_identifier and every app class (in units of standard deviations)
        std_distance = 1 - np.absolute(self.app_qosa_mapping[self.scores] - io_identifier[self.scores].to_numpy())
        #print(std_distance)
        #Get the magnitude of the fitnesses
        std_distance.loc[:,"magnitude"] = self.get_magnitude(std_distance)
        #Add qosas to dataframe
        #std_distance.loc[:,"qosas"] = self.app_classes["qosas"].to_numpy()
        #Add features
        std_distance.loc[:,self.features] = self.app_qosa_mapping[self.features].to_numpy()
        return std_distance
