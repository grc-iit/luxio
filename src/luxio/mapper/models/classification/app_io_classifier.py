import sys,os
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from .behavior_classifier import BehaviorClassifier
from luxio.common.configuration_manager import *
from luxio.mapper.models.common import *
from luxio.mapper.models.regression.forest.random_forest import RFERandomForestRegressor
from luxio.mapper.models.metrics import r2Metric, RelativeAccuracyMetric, RelativeErrorMetric
from luxio.mapper.models.transforms.transformer_factory import TransformerFactory
from luxio.mapper.models.transforms.log_transformer import LogTransformer
from luxio.mapper.models.transforms.chain_transformer import ChainTransformer
from luxio.mapper.models.dimensionality_reduction.dimension_reducer_factory import DimensionReducerFactory

from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.metrics import davies_bouldin_score
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

import itertools

import pprint, warnings
pp = pprint.PrettyPrinter(depth=6)
pd.options.mode.chained_assignment = None

class AppClassifier(BehaviorClassifier):
    def __init__(self, features, mandatory_features, output_vars, score_conf, dataset_path, random_seed=132415, n_jobs=6):
        super().__init__(features, mandatory_features, output_vars, score_conf, dataset_path, random_seed, n_jobs)
        self.app_classes = None #A pandas dataframe containing: means, stds, number of entries, and sslos
        self.thresh = .25
        self.app_sslo_mapping = None
        self.scores = None
        self.sslo_scores = None

    def feature_selector(self, X, y):
        train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=.1, random_state=self.random_seed)
        param_grid = {
            'n_features': [15, 20, 25],
            'max_depth': [5, 10],
            #'transform_method': [None, 'log10p1'],
            'reducer_method': ['random-forest']
        }
        print("Heuristic Feature Reduction")
        heuristic_reducer = DimensionReducerFactory(features=self.features, n_jobs=self.n_jobs)
        heuristic_reducer.fit(X,y)
        print("Fitting Feature Reduced Model")
        model = RFERandomForestRegressor(
            features=self.features,
            #transform=TransformerFactory(),
            transform_y=LogTransformer(add=1,base=10),
            heuristic_reducer=heuristic_reducer,
            n_features_heur=35,
            fitness_metric=RelativeAccuracyMetric(),
            error_metric=RelativeErrorMetric())
        search = GridSearchCV(model, param_grid, cv=KFold(n_splits=5, random_state=self.random_seed, shuffle=True), n_jobs=self.n_jobs, verbose=2)
        search.fit(train_x, train_y)
        self.feature_selector_ = search.best_estimator_
        self.feature_importances_ = self.feature_selector_.feature_importances_
        self.features_ = self.feature_selector_.features_
        self.named_feature_importances_ = pd.DataFrame([(feature,importance) for feature,importance in zip(self.features_, self.feature_importances_)])

    def fit(self, X:pd.DataFrame, k=8):
        """
        Identify groups of application behavior from a dataset of traces using the features.
        Calculate a standardized set of scores that are common between the apps and sslos
        """
        #self.named_feature_importances_ = pd.DataFrame(
        #    [(feature, importance) for feature, importance in zip(self.features_, self.feature_importances_)])
        #Identify clusters of transformed data
        self.transform_ = ChainTransformer([LogTransformer(base=10,add=1), MinMaxScaler()])
        X_features = self.transform_.fit_transform(X[self.features_])*self.feature_importances_
        if k is None:
            for k in [2, 4, 8, 10, 15]:
                self.model_ = KMeans(n_clusters=k)
                self.labels_ = self.model_.fit_predict(X_features)
                print(f"SCORE k={k}: {self.score(X_features, self.labels_)} {self.model_.inertia_}")
            k = int(input("Optimal k: "))
        self.model_ = KMeans(n_clusters=k)
        self.labels_ = self.model_.fit_predict(X_features)
        #Cluster non-transformed data
        self.app_classes = self.standardize(X)
        self.app_classes = self._create_groups(self.app_classes, self.labels_).reset_index()
        self.app_classes['app_id'] = self.app_classes.index
        print(self.app_classes)
        return self

    def score(self, X:pd.DataFrame, labels:np.array) -> float:
        return davies_bouldin_score(X, labels)

    def standardize(self, io_identifier):
        return io_identifier
        #Define the scoring categories
        SCORES = self.score_conf
        #Get score weights and remember the score categories
        if self.scores is None:
            self.scores = list(SCORES.keys())
            self.score_weights = []
            for features in SCORES.values():
                features = self.named_feature_importances_.columns.intersection(features)
                self.score_weights.append(self.named_feature_importances_[features].to_numpy().sum())
            self.score_weights = pd.Series(self.score_weights, index=self.scores) / np.sum(self.score_weights)

        #Normalize the IOID to the range [0,1] and scale by feature importance
        scaled_features = pd.DataFrame(self.transform_.transform(io_identifier[self.features].astype(float)), columns=self.features_)
        for score_name,features,score_weight in zip(SCORES.keys(), SCORES.values(), self.score_weights.to_numpy()):
            features = scaled_features.columns.intersection(features)
            if score_name == 'SEQUENTIALITY' and "TOTAL_IO_OPS" in io_identifier.columns:
                io_identifier.loc[:,score_name] = io_identifier[features].sum(axis=1).to_numpy()/io_identifier['TOTAL_IO_OPS'].to_numpy()
            else:
                io_identifier.loc[:,score_name] = (scaled_features[features] * self.named_feature_importances_[features].to_numpy()).sum(axis=1).to_numpy()/score_weight

        return io_identifier

    def define_low_med_high(self, dir):
        SCORES = self.score_conf
        n = len(self.features)
        scaled_features = pd.DataFrame([[size]*n for size in [.33, .66, 1]], columns=self.features)
        unscaled_features = pd.DataFrame(self.transform_.inverse_transform(scaled_features), columns=self.features)
        for score_name,features,score_weight in zip(SCORES.keys(), SCORES.values(), self.score_weights):
            features = self.feature_importances_.columns.intersection(features)
            unscaled_features.loc[:,score_name] = (unscaled_features[features] * self.feature_importances_[features].to_numpy()).sum(axis=1).to_numpy()/score_weight
        unscaled_features[self.scores] = LogTransformer(base=10,add=1).transform(unscaled_features[self.scores]).astype(int)
        unscaled_features[self.scores] = "10^" + unscaled_features[self.scores].astype(str)
        unscaled_features[self.scores].to_csv(os.path.join(dir, "low_med_high.csv"))

    def analyze_classes(self, dir=None):
        if dir is not None:
            self.define_low_med_high(dir)
            app_classes = self.app_classes.copy()
            #Save set of appclasses
            app_classes[self.features + self.scores + ["count"]].transpose().to_csv(os.path.join(dir, "orig_behavior_means.csv"))
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
            app_classes = app_classes[self.scores + self.features + ["count"]]
            app_classes = app_classes.groupby(self.scores + self.features).sum().reset_index()
            app_classes.sort_values("count", ascending=False, inplace=True)
            app_classes[self.scores + ["count"]].transpose().to_csv(os.path.join(dir, "behavior_means.csv"))

    def visualize(self, df, path=None):
        df = self.standardize(df)
        PERFORMANCE = ["TOTAL_READ_TIME", "TOTAL_WRITE_TIME", "TOTAL_MD_TIME"]
        train_x, test_x, train_y, test_y = train_test_split(df[PERFORMANCE], self.labels_, test_size=5000, stratify=self.labels_)
        for lr in [200]:
            for perplexity in [30, 50]:
                print(f"PERPLEXITY: {perplexity}")
                X = TSNE(n_components=2, perplexity=perplexity, learning_rate=lr, n_jobs=6).fit_transform(test_x)
                plt.scatter(X[:,0], X[:,1], label=test_y, c=test_y, alpha=.3)
                plt.show()
                if path is not None:
                    plt.savefig(path)
                plt.close()

    def filter_sslos(self, storage_classifier):
        """
        For each application class, filter out sslos that have little chance of being useful.
        """
        sslos = []
        app_classes = []
        #Compare every application class with every sslo
        for idx,app_class in self.app_classes.iterrows():
            coverages = storage_classifier.get_coverages(app_class)
            coverages = coverages[coverages.magnitude >= self.conf.min_coverage_thresh]
            coverages['app_id'] = app_class['app_id'].astype(int)
            sslos.append(coverages)
        #Add the sslos to the dataframe
        self.app_sslo_mapping = pd.concat(sslos)
        print(self.app_sslo_mapping)

    def get_magnitude(self, fitness:pd.DataFrame):
        """
        Convert the fitness vector into a single score.
        """
        scores = self.app_classes[self.scores].columns.intersection(fitness.columns)
        fitness = fitness.fillna(0)
        fitness[fitness[scores] > 1] = 1
        magnitude = (fitness[scores].to_numpy()*self.score_weights[scores].to_numpy()).sum(axis=1)/self.score_weights[scores].sum()
        return magnitude

    def get_fitnesses(self, io_identifier:pd.DataFrame) -> pd.DataFrame:
        """
        Determine how well the I/O Identifier fits within each class of behavior
        This assumes the io_identifier has already been standardized
        """
        scores = self.scores
        #Get the fitness between io_identifier and every app class (score)
        fitness = 1 - np.abs(app_classes[scores] - io_identifier[scores].to_numpy())
        #Add features
        fitness.loc[:,self.features] = app_classes[self.features].to_numpy()
        fitness.loc[:,'app_id'] = app_classes['app_id']
        #Get the magnitude of the fitnesses
        fitness.loc[:,"magnitude"] = self.get_magnitude(fitness)
        return fitness
