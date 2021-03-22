import sys,os
import pickle as pkl
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from .behavior_classifier import BehaviorClassifier
from luxio.common.configuration_manager import *

from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from clever.transformers import *
from clever.models.cluster import KMeans
from pyclustering.cluster.kmedoids import kmedoids

from sklearn.metrics import davies_bouldin_score

import pprint, warnings
pp = pprint.PrettyPrinter(depth=6)
pd.options.mode.chained_assignment = None

class StorageClassifier(BehaviorClassifier):
    def __init__(self, features:list, score_conf:dict, feature_importances:pd.DataFrame, mandatory_features:List[str]=None):
        super().__init__(features, score_conf, feature_importances, mandatory_features)
        self.qosas = None #A dataframe containing: means, stds, ns

    def score(self, X:pd.DataFrame, labels:np.array) -> float:
        return davies_bouldin_score(X, labels)

    def fit(self, X:pd.DataFrame=None):
        X = X.drop_duplicates()
        #Identify clusters of transformed data
        #self.transform_ = ChainTransformer([LogTransformer(base=10,add=1), MinMaxScaler()])
        self.transform_ = MinMaxScaler()
        X_features = self.transform_.fit_transform(X[self.features])*self.feature_importances.max(axis=1).to_numpy()
        """
        for k in [4, 6, 8, 10, 15, 20, 30, 50, 100, 150, 300]:
            self.model_ = KMeans(k=k)
            self.labels_ = self.model_.fit_predict(X_features)
            print(f"SCORE k={k}: {self.score(X_features, self.labels_)} {self.model_.km_.inertia_}")
        sys.exit(1)
        """
        self.model_ = KMeans(k=20)
        self.labels_ = self.model_.fit_predict(X_features)
        #Cluster non-transformed data
        self.qosas = self.standardize(X)
        self.qosas = self._create_groups(self.qosas, self.labels_, other=self.mandatory_features)
        #TODO: Classify deployments into QoSAs 100% correctly...
        self.qosas.rename(columns={"labels":"qosa_id"}, inplace=True)
        self.qosa_to_deployment = X
        self.qosa_to_deployment.loc[:,"qosa_id"] = self.labels_
        return self

    def define_low_med_high(self, dir):
        SCORES = self.score_conf
        n = len(self.features)
        scaled_features = pd.DataFrame([[size]*n for size in [.33, .66, 1]], columns=self.features)
        unscaled_features = pd.DataFrame(self.transform_.inverse_transform(scaled_features), columns=self.features)
        for score_name,features,score_weight in zip(SCORES.keys(), SCORES.values(), self.score_weights):
            features = self.feature_importances.columns.intersection(features)
            unscaled_features.loc[:,score_name] = (unscaled_features[features] * self.feature_importances[features].to_numpy()).sum(axis=1).to_numpy()/score_weight
        unscaled_features[self.scores].to_csv(os.path.join(dir, "low_med_high.csv"))

    def analyze(self, dir=None):
        if dir is not None:
            #trans = ChainTransformer([LogTransformer(base=10,add=1), MinMaxScaler()]).fit(self.qosa_to_deployment)
            self.define_low_med_high(dir)
            qosas = self.qosas.copy()
            #Apply standardization
            qosas[self.scores].to_csv(os.path.join(dir, "orig_behavior_means.csv"))
            #Apply transformation to features
            qosas.loc[:,self.features] = (self.transform_.transform(qosas[self.features])*3).astype(int)
            #Apply transformation to scores
            qosas.loc[:,self.scores] = (qosas[self.scores]*3).fillna(0).astype(int)
            #Label each bin
            for feature in self.features + self.scores:
                for i,label in enumerate(["low", "medium", "high"]):
                    qosas.loc[qosas[feature] == i,feature] = label
                qosas.loc[qosas[feature] == 3,feature] = "high"
            #Store the application classes
            qosas = qosas[self.scores + self.features + self.mandatory_features + ["count"]]
            qosas = qosas.groupby(self.scores + self.mandatory_features).sum().reset_index()
            qosas.sort_values("count", ascending=False, inplace=True)
            qosas[self.scores + self.mandatory_features + ["count"]].to_csv(os.path.join(dir, "behavior_means.csv"))

    def standardize(self, qosas:pd.DataFrame):
        SCORES = self.score_conf
        #Get score weights and remember the score categories
        if self.scores is None:
            self.scores = list(SCORES.keys())
            self.score_weights = []
            for features in SCORES.values():
                features = self.feature_importances.columns.intersection(features)
                self.score_weights.append(self.feature_importances[features].to_numpy().sum())
            self.score_weights = pd.Series(self.score_weights, index=self.scores) / np.sum(self.score_weights)

        scaled_features = pd.DataFrame(self.transform_.transform(qosas[self.features].astype(float)), columns=self.features)
        for score_name,features,score_weight in zip(SCORES.keys(), SCORES.values(), self.score_weights):
            features = scaled_features.columns.intersection(features)
            qosas.loc[:,score_name] = (scaled_features[features] * self.feature_importances[features].to_numpy()).sum(axis=1).to_numpy()/score_weight

        return qosas

    def get_magnitude(self, coverage:pd.DataFrame):
        scores = self.qosas[self.scores].columns.intersection(coverage.columns)
        coverage = coverage.fillna(0)
        coverage[coverage[scores] > 1] = 1
        return ((coverage[scores]*self.score_weights).sum(axis=1)/np.sum(self.score_weights)).to_numpy()

    def get_coverages(self, io_identifier:pd.DataFrame, qosas:pd.DataFrame=None) -> pd.DataFrame:
        """
        Get the extent to which an qosas is covered by each QoSA
        qosas: Either the centroid of an app class or the signature of a unique application
        """
        if qosas is None:
            qosas = self.qosas
        #Filter out incompatible qosas (TODO: fix this hack, interface was float for some reason...)
        qosas['interface'].astype(int, copy=False)
        interface = int(io_identifier['INTERFACE'])
        mandatory = (qosas[['interface']].to_numpy() & interface) == interface
        qosas = qosas[mandatory]
        #Get the coverage between qosas and every qosa
        coverage = 1 - (qosas[self.scores] - io_identifier[self.scores].to_numpy())
        #Add features
        coverage.loc[:,self.features] = qosas[self.features].to_numpy()
        coverage.loc[:,'qosa_id'] = qosas['qosa_id']
        #Get the magnitude of the fitnesses
        coverage.loc[:,"magnitude"] = self.get_magnitude(coverage)
        return coverage
