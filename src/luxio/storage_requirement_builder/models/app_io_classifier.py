import sys,os
from .behavior_classifier import BehaviorClassifier
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
from sklearn.metrics import davies_bouldin_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from clever.transformers import *
from clever.models.cluster import KMeans

import itertools

import pprint, warnings
pp = pprint.PrettyPrinter(depth=6)
pd.options.mode.chained_assignment = None

class AppClassifier(BehaviorClassifier):
    def __init__(self, feature_importances:pd.DataFrame, mandatory_features:List[str]=None):
        super().__init__(feature_importances, mandatory_features)
        self.app_classes = None #A pandas dataframe containing: means, stds, number of entries, and qosas
        self.thresh = .25
        self.app_qosa_mapping = None
        self.qosas = None
        self.scores = None

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
        #self.transform_ = ChainTransformer([RobustScaler(), MinMaxScaler()])
        X_features = self.transform_.fit_transform(X[self.features])*self.feature_importances.max(axis=1).to_numpy()
        #for k in [4, 6, 8, 10, 15, 30, 50]:
        #    self.model_ = KMeans(k=k)
        #    self.labels_ = self.model_.fit_predict(X_features)
        #    print(f"SCORE k={k}: {self.score(X_features, self.labels_)} {self.model_.km_.inertia_}")
        self.model_ = KMeans(k=10)
        self.labels_ = self.model_.fit_predict(X_features)
        #Cluster non-transformed data
        self.app_classes = self.standardize(X)
        self.app_classes = self._create_groups(self.app_classes, self.labels_, other=self.mandatory_features)
        #self.app_classes = self.standardize(self.app_classes)
        print(self.app_classes)
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

    def analyze(self, dir=None):
        if dir is not None:
            app_classes = self.app_classes.copy()
            #Apply standardization
            app_classes[self.scores].to_csv(os.path.join(dir, "orig_behavior_means.csv"))
            #Apply transformation to features
            app_classes.loc[:,self.features] = (self.transform_.transform(app_classes[self.features])*3).astype(int)
            #Apply transformation to scores
            app_classes.loc[:,self.scores] = (app_classes[self.scores]/self.score_weights*3).fillna(0).astype(int)
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

    """
    def visualize(self, df, path=None):
        df = self.standardize(df)
        PERFORMANCE = ["TOTAL_READ_TIME", "TOTAL_WRITE_TIME", "TOTAL_MD_TIME"]
        train_x, test_x, train_y, test_y = train_test_split(np.log(df[PERFORMANCE]+1), self.labels_, test_size=50000, stratify=self.labels_)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel("TOTAL_READ_TIME")
        ax.set_ylabel("TOTAL_WRITE_TIME")
        ax.set_zlabel("TOTAL_MD_TIME")
        ax.scatter(test_x.to_numpy()[:,0], test_x.to_numpy()[:,1], test_x.to_numpy()[:,2], label=test_y, c=test_y, alpha=.3)
        plt.show()
        if path is not None:
            plt.savefig(path)
        plt.close()
    """

    def standardize(self, io_identifier, weighted=True):
        #Define the scoring categories
        SCORES = {
            "SMALL_READ" : [
                "TOTAL_POSIX_SIZE_READ_0_100", "TOTAL_POSIX_SIZE_READ_100_1K",
                "TOTAL_MPIIO_SIZE_READ_AGG_0_100", "TOTAL_MPIIO_SIZE_READ_AGG_100_1K"
            ],
            "MEDIUM_READ" : [
                "TOTAL_POSIX_SIZE_READ_1K_10K", "TOTAL_POSIX_SIZE_READ_10K_100K", "TOTAL_POSIX_SIZE_READ_100K_1M",
                "TOTAL_MPIIO_SIZE_READ_AGG_1K_10K", "TOTAL_MPIIO_SIZE_READ_AGG_10K_100K", "TOTAL_MPIIO_SIZE_READ_AGG_100K_1M"
            ],
            "LARGE_READ" : [
                "TOTAL_POSIX_SIZE_READ_1M_4M", "TOTAL_POSIX_SIZE_READ_4M_10M", "TOTAL_POSIX_SIZE_READ_10M_100M", "TOTAL_POSIX_SIZE_READ_100M_1G", "TOTAL_POSIX_SIZE_READ_1G_PLUS",
                "TOTAL_MPIIO_SIZE_READ_AGG_1M_4M", "TOTAL_MPIIO_SIZE_READ_AGG_4M_10M", "TOTAL_MPIIO_SIZE_READ_AGG_10M_100M", "TOTAL_MPIIO_SIZE_READ_AGG_100M_1G", "TOTAL_MPIIO_SIZE_READ_AGG_1G_PLUS",
            ],
            "SMALL_WRITE" : [
                "TOTAL_POSIX_SIZE_WRITE_0_100", "TOTAL_POSIX_SIZE_WRITE_100_1K",
                "TOTAL_MPIIO_SIZE_WRITE_AGG_0_100", "TOTAL_MPIIO_SIZE_WRITE_AGG_100_1K"
            ],
            "MEDIUM_WRITE" : [
                "TOTAL_POSIX_SIZE_WRITE_1K_10K", "TOTAL_POSIX_SIZE_WRITE_10K_100K", "TOTAL_POSIX_SIZE_WRITE_100K_1M",
                "TOTAL_MPIIO_SIZE_WRITE_AGG_1K_10K", "TOTAL_MPIIO_SIZE_WRITE_AGG_10K_100K", "TOTAL_MPIIO_SIZE_WRITE_AGG_100K_1M"
            ],
            "LARGE_WRITE" : [
                "TOTAL_POSIX_SIZE_WRITE_1M_4M", "TOTAL_POSIX_SIZE_WRITE_4M_10M", "TOTAL_POSIX_SIZE_WRITE_10M_100M", "TOTAL_POSIX_SIZE_WRITE_100M_1G", "TOTAL_POSIX_SIZE_WRITE_1G_PLUS",
                "TOTAL_MPIIO_SIZE_WRITE_AGG_1M_4M", "TOTAL_MPIIO_SIZE_WRITE_AGG_4M_10M", "TOTAL_MPIIO_SIZE_WRITE_AGG_10M_100M", "TOTAL_MPIIO_SIZE_WRITE_AGG_100M_1G", "TOTAL_MPIIO_SIZE_WRITE_AGG_1G_PLUS"
            ],
            "SMALL_IO" : [
                "TOTAL_SIZE_IO_0_100", "TOTAL_SIZE_IO_100_1K", "SMALL_READ", "SMALL_WRITE"
            ],
            "MEDIUM_IO" : [
                "TOTAL_SIZE_IO_1K_10K", "TOTAL_SIZE_IO_10K_100K", "TOTAL_SIZE_IO_100K_1M", "MEDIUM_READ", "MEDIUM_WRITE"
            ],
            "LARGE_IO" : [
                "TOTAL_SIZE_IO_1M_4M", "TOTAL_SIZE_IO_4M_10M", "TOTAL_SIZE_IO_10M_100M", "TOTAL_SIZE_IO_100M_1G", "TOTAL_SIZE_IO_1G_PLUS", "LARGE_READ", "LARGE_WRITE"
            ],
            "BYTES_READ" : ["TOTAL_BYTES_READ"],
            "BYTES_WRITTEN" : ["TOTAL_BYTES_WRITTEN"],
            "MD_HEAVINESS" : ["TOTAL_MD_OPS", "TOTAL_STDIO_OPENS", "TOTAL_POSIX_OPENS", "TOTAL_MPIIO_COLL_OPENS", "TOTAL_MPIIO_INDEP_OPENS", "TOTAL_POSIX_STATS", "TOTAL_POSIX_FDSYNCS", "TOTAL_MPIIO_SYNCS", "TOTAL_STDIO_SEEKS", "TOTAL_POSIX_SEEKS"],
            "READ_HEAVINESS" : ["TOTAL_READ_OPS", "TOTAL_MPIIO_COLL_READS", "TOTAL_MPIIO_INDEP_READS", "TOTAL_MPIIO_SPLIT_READS", "TOTAL_POSIX_READS", "TOTAL_STDIO_READS"],
            "WRITE_HEAVINESS" : ["TOTAL_WRITE_OPS", "TOTAL_MPIIO_COLL_WRITES", "TOTAL_MPIIO_INDEP_WRITES", "TOTAL_MPIIO_SPLIT_WRITES", "TOTAL_POSIX_WRITES", "TOTAL_STDIO_WRITES"],
            "SEQUENTIALITY" : ["TOTAL_POSIX_SEQ_READS", "TOTAL_POSIX_SEQ_WRITES", "TOTAL_POSIX_CONSEC_READS", "TOTAL_POSIX_CONSEC_WRITES", "TOTAL_MPIIO_COLL_READS", "TOTAL_MPIIO_COLL_WRITES"],
            "SCALE" : ["NPROCS"]
        }

        #Get score weights and remember the score categories
        if self.scores is None:
            self.scores = list(SCORES.keys())
            self.score_weights = []
            for features in SCORES.values():
                features = self.feature_importances.columns.intersection(features)
                self.score_weights.append(self.feature_importances[features].to_numpy().sum())
            self.score_weights = np.array(self.score_weights)

        #Normalize the IOID to the range [0,1] and scale by feature importance
        scaled_features = pd.DataFrame(self.transform_.transform(io_identifier[self.features].astype(float)), columns=self.features)
        for score_name,features in SCORES.items():
            features = scaled_features.columns.intersection(features)
            if weighted:
                io_identifier.loc[:,score_name] = (scaled_features[features] * self.feature_importances[features].to_numpy()).sum(axis=1).to_numpy()
            else:
                io_identifier.loc[:,score_name] = io_identifier[features].sum(axis=1).to_numpy()
            if score_name == 'SEQUENTIALITY':
                io_identifier.loc[:,score_name] = np.sum(self.feature_importances[features].to_numpy()) * io_identifier[features].sum(axis=1).to_numpy()/io_identifier['TOTAL_IO_OPS'].to_numpy()

        return io_identifier

    def get_magnitude(self, fitness:pd.DataFrame):
        """
        Convert the fitness vector into a single score.
        Some features are mandatory, and will cause fitness to be 0 if not met.
        Some features are continuous, and have a spectrum of values
        """
        fitness.loc[:, self.mandatory_features] = 0
        thresh = ((fitness[self.scores]*self.score_weights).sum(axis=1)/np.sum(self.score_weights)).to_numpy()
        return mandatory * thresh

    def get_fitnesses(self, io_identifier:pd.DataFrame) -> pd.DataFrame:
        """
        Determine how well the I/O Identifier fits within each class of behavior
        """
        #Calculate the scores
        io_identifier = self.standardize(io_identifier)
        #Get the distance between io_identifier and every app class
        distance = 1 - np.absolute(self.app_qosa_mapping[self.scores] - io_identifier[self.scores].to_numpy())
        #Get the magnitude of the fitnesses
        distance.loc[:,"magnitude"] = self.get_magnitude(distance)
        #Add features
        distance.loc[:,self.features] = self.app_qosa_mapping[self.features].to_numpy()
        return distance
