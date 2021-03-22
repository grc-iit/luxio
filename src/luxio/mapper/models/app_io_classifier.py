import sys,os
from .behavior_classifier import BehaviorClassifier
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
from sklearn.metrics import davies_bouldin_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from luxio.common.configuration_manager import *

from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from clever.transformers import *
from clever.models.cluster import KMeans

import itertools

import pprint, warnings
pp = pprint.PrettyPrinter(depth=6)
pd.options.mode.chained_assignment = None

class AppClassifier(BehaviorClassifier):
    def __init__(self, features:list, score_conf:dict, feature_importances:pd.DataFrame=None, mandatory_features:List[str]=None):
        super().__init__(features, score_conf, feature_importances, mandatory_features)
        self.app_classes = None #A pandas dataframe containing: means, stds, number of entries, and qosas
        self.thresh = .25
        self.app_qosa_mapping = None
        self.scores = None
        self.qosa_scores = None

    def score(self, X:pd.DataFrame, labels:np.array) -> float:
        return davies_bouldin_score(X, labels)

    def fit(self, X:pd.DataFrame):
        """
        Identify groups of application behavior from a dataset of traces using the features.
        Calculate a standardized set of scores that are common between the apps and QoSAs
        """
        #X = X.iloc[0:100,:]
        #Identify clusters of transformed data
        self.transform_ = ChainTransformer([LogTransformer(base=10,add=1), MinMaxScaler()])
        X_features = self.transform_.fit_transform(X[self.features])*self.feature_importances.max(axis=1).to_numpy()
        #for k in [2, 4, 6, 8, 10, 12, 15, 18, 20, 25, 30, 35, 40, 45, 50]:
        #    self.model_ = KMeans(k=k)
        #    self.labels_ = self.model_.fit_predict(X_features)
        #    print(f"SCORE k={k}: {self.score(X_features, self.labels_)} {self.model_.km_.inertia_}")
        #sys.exit(1)
        self.model_ = KMeans(k=12)
        self.labels_ = self.model_.fit_predict(X_features)
        #Cluster non-transformed data
        self.app_classes = self.standardize(X)
        self.app_classes = self._create_groups(self.app_classes, self.labels_, other=self.mandatory_features).reset_index()
        self.app_classes['app_id'] = self.app_classes.index
        #self.app_classes = self.standardize(self.app_classes)
        print(self.app_classes)
        return self

    def filter_qosas(self, storage_classifier):
        """
        For each application class, filter out QoSAs that have little chance of being useful.
        """
        qosas = []
        app_classes = []
        #Compare every application class with every qosa
        for idx,app_class in self.app_classes.iterrows():
            coverages = storage_classifier.get_coverages(app_class)
            coverages = coverages[coverages.magnitude >= self.conf.min_coverage_thresh]
            print(coverages)
            coverages['app_id'] = app_class['app_id'].astype(int)
            qosas.append(coverages)
        #Add the qosas to the dataframe
        self.app_qosa_mapping = pd.concat(qosas)
        print(self.app_qosa_mapping)

    def define_low_med_high(self, dir):
        SCORES = self.score_conf
        n = len(self.features)
        scaled_features = pd.DataFrame([[size]*n for size in [.33, .66, 1]], columns=self.features)
        unscaled_features = pd.DataFrame(self.transform_.inverse_transform(scaled_features), columns=self.features)
        for score_name,features,score_weight in zip(SCORES.keys(), SCORES.values(), self.score_weights):
            features = self.feature_importances.columns.intersection(features)
            unscaled_features.loc[:,score_name] = (unscaled_features[features] * self.feature_importances[features].to_numpy()).sum(axis=1).to_numpy()/score_weight
        unscaled_features[self.scores] = LogTransformer(base=10,add=1).transform(unscaled_features[self.scores]).astype(int)
        unscaled_features[self.scores] = "10^" + unscaled_features[self.scores].astype(str)
        unscaled_features[self.scores].to_csv(os.path.join(dir, "low_med_high.csv"))

    def analyze(self, dir=None):
        if dir is not None:
            self.define_low_med_high(dir)
            app_classes = self.app_classes.copy()
            #Apply standardization
            app_classes[self.scores].to_csv(os.path.join(dir, "orig_behavior_means.csv"))
            #Apply transformation to features
            app_classes.loc[:,self.features] = (self.transform_.transform(app_classes[self.features])*3).astype(int)
            #Apply transformation to scores
            app_classes.loc[:,self.scores] = (app_classes[self.scores]*3).fillna(0).astype(int)
            #Label each bin
            for feature in self.features + self.scores:
                for i,label in enumerate(["low", "medium", "high"]):
                    app_classes.loc[app_classes[feature] == i,feature] = label
                app_classes.loc[app_classes[feature] == 3,feature] = "high"
            #Store the application classes
            app_classes = app_classes[self.scores + self.features + self.mandatory_features + ["count"]]
            app_classes = app_classes.groupby(self.scores + self.features + self.mandatory_features).sum().reset_index()
            app_classes.sort_values("count", ascending=False, inplace=True)
            app_classes[self.scores + self.mandatory_features + ["count"]].to_csv(os.path.join(dir, "behavior_means.csv"))

    def visualize(self, df, path=None):
        df = self.standardize(df)
        PERFORMANCE = ["TOTAL_READ_TIME", "TOTAL_WRITE_TIME", "TOTAL_MD_TIME"]
        train_x, test_x, train_y, test_y = train_test_split(df[PERFORMANCE], self.labels_, test_size=50000, stratify=self.labels_)
        for lr in [200]:
            for perplexity in [30, 50]:
                print(f"PERPLEXITY: {perplexity}")
                X = TSNE(n_components=2, perplexity=perplexity, learning_rate=lr, n_jobs=6).fit_transform(test_x)
                plt.scatter(X[:,0], X[:,1], label=test_y, c=test_y, alpha=.3)
                plt.show()
                if path is not None:
                    plt.savefig(path)
                plt.close()

    def standardize(self, io_identifier):
        #Define the scoring categories
        SCORES = self.score_conf
        #Get score weights and remember the score categories
        if self.scores is None:
            self.scores = list(SCORES.keys())
            self.score_weights = []
            for features in SCORES.values():
                features = self.feature_importances.columns.intersection(features)
                self.score_weights.append(self.feature_importances[features].to_numpy().sum())
            self.score_weights = pd.Series(self.score_weights, index=self.scores) / np.sum(self.score_weights)

        #Normalize the IOID to the range [0,1] and scale by feature importance
        scaled_features = pd.DataFrame(self.transform_.transform(io_identifier[self.features].astype(float)), columns=self.features)
        for score_name,features,score_weight in zip(SCORES.keys(), SCORES.values(), self.score_weights.to_numpy()):
            features = scaled_features.columns.intersection(features)
            if score_name == 'SEQUENTIALITY' and "TOTAL_IO_OPS" in io_identifier.columns:
                io_identifier.loc[:,score_name] = io_identifier[features].sum(axis=1).to_numpy()/io_identifier['TOTAL_IO_OPS'].to_numpy()
            else:
                io_identifier.loc[:,score_name] = (scaled_features[features] * self.feature_importances[features].to_numpy()).sum(axis=1).to_numpy()/score_weight

        return io_identifier

    def get_magnitude(self, fitness:pd.DataFrame):
        """
        Convert the fitness vector into a single score.
        """
        scores = self.app_classes[self.scores].columns.intersection(fitness.columns)
        fitness = fitness.fillna(0)
        fitness[fitness[scores] > 1] = 1
        magnitude = ((fitness[scores].to_numpy()*self.score_weights[scores].to_numpy()).sum(axis=1)/self.score_weights[scores].sum())
        return magnitude

    def get_fitnesses(self, io_identifier:pd.DataFrame) -> pd.DataFrame:
        """
        Determine how well the I/O Identifier fits within each class of behavior
        """
        scores = self.scores
        #Calculate the scores
        io_identifier = self.standardize(io_identifier)
        #Filter out incompatible app classes
        mandatory = (io_identifier[self.mandatory_features].to_numpy() & self.app_classes[self.mandatory_features].to_numpy()) == io_identifier[self.mandatory_features].to_numpy()
        app_classes = self.app_classes[mandatory]
        #Get the fitness between io_identifier and every app class (score)
        fitness = 1 - (app_classes[scores] - io_identifier[scores].to_numpy())
        #Add features
        fitness.loc[:,self.features] = app_classes[self.features].to_numpy()
        fitness.loc[:,'app_id'] = app_classes['app_id']
        #Get the magnitude of the fitnesses
        fitness.loc[:,"magnitude"] = self.get_magnitude(fitness)
        return fitness
