import sys,os
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np

#fitness(io_identity, app_class) -> fitness
#coverage(app_class, slo_id) -> Ranking
#coverage: <Bytes Read, Bytes Written>




from .behavior_classifier import BehaviorClassifier
from luxio.common.configuration_manager import *
from luxio.mapper.models.common import *
from luxio.mapper.models.regression.forest.random_forest import RFERandomForestRegressor
from luxio.mapper.models.metrics import r2Metric, RelativeAccuracyMetric, RelativeErrorMetric
from luxio.mapper.models.transforms.transformer_factory import TransformerFactory
from luxio.mapper.models.transforms.log_transformer import LogTransformer
from luxio.mapper.models.transforms.chain_transformer import ChainTransformer
from luxio.mapper.models.dimensionality_reduction.dimension_reducer_factory import DimensionReducerFactory, RandomForestReducer

from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.metrics import davies_bouldin_score
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans, SpectralClustering
import copy
import itertools

import pprint, warnings
pp = pprint.PrettyPrinter(depth=6)
pd.options.mode.chained_assignment = None

class AppClassifier(BehaviorClassifier):
    def __init__(self, features, mandatory_features, output_vars, score_conf, dataset_path, random_seed=132415, n_jobs=6):
        super().__init__(features, mandatory_features, output_vars, score_conf, dataset_path, random_seed, n_jobs)
        self.app_classes_ = None #A pandas dataframe containing: means, stds, number of entries, and sslos
        self.thresh = .25
        self.app_sslo_mapping = None
        self.scores = None
        self.sslo_scores = None

    def feature_selector(self, X, y):
        train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=.1, random_state=self.random_seed)
        param_grid = {
            'n_features': [20, 25],
            'max_depth': [5, 8, 10],
            'n_estimators': [10, 15],
            'reducer_method': ['random-forest']
        }
        print("Heuristic Feature Reduction")
        heuristic_reducer = DimensionReducerFactory(features=self.features, n_jobs=self.n_jobs, random_seed=self.random_seed)
        heuristic_reducer.fit(X,y)
        print("Fitting Feature Reduced Model")
        model = RFERandomForestRegressor(
            features=self.features,
            transform_y=LogTransformer(add=1,base=10),
            heuristic_reducer=heuristic_reducer,
            n_features_heur=50,
            fitness_metric=RelativeAccuracyMetric(add=10),
            error_metric=RelativeErrorMetric(add=10))
        search = GridSearchCV(model, param_grid, cv=KFold(n_splits=5, random_state=self.random_seed, shuffle=True), n_jobs=self.n_jobs, verbose=2)
        search.fit(train_x, train_y)
        self.feature_selector_ = search.best_estimator_
        self.feature_importances_ = self.feature_selector_.feature_importances_
        self.features_ = self.feature_selector_.features_
        self.named_feature_importances_ = pd.DataFrame([(feature,importance) for feature,importance in zip(self.features_, self.feature_importances_)])

    def fit(self, X, k=None, score_conf=None):
        """
        Identify groups of application behavior from a dataset of traces using the features.
        Calculate a standardized set of scores that are common between the apps and sslos
        """

        #Initialize the standardization method
        if score_conf is not None:
            self.score_conf = score_conf
        self.cluster_features_ = list(self.features_) + list(self.output_vars)
        self.score_features_ = list(self.features) + list(self.output_vars)
        self.score_default_feature_weights_ = pd.DataFrame(
            dict([(feature, weight) for feature, weight in zip(self.features_, self.feature_importances_)] +
                 [(feature, 0 / len(self.output_vars)) for feature in self.output_vars]), index=[0]
        )
        self.transform_ = ChainTransformer([LogTransformer(base=10, add=1), MinMaxScaler()]).fit(X[self.score_features_])
        self._init_scoring()
        #Create application classes
        self.app_classes_ = self.standardize(X)
        X_features = self.app_classes_[self.score_features_]
        X_features.loc[:,self.score_features_] = self.transform_.transform(X_features)
        X_features = X_features[self.cluster_features_]
        if k is None:
            for k in [2, 4, 6, 8, 10, 15]:
                self.model_ = KMeans(n_clusters=k)
                self.labels_ = self.model_.fit_predict(X_features)
                print(f"SCORE k={k}: {self.score(X_features, self.labels_)} {self.model_.inertia_}")
            k = int(input("Optimal k: "))
        self.model_ = KMeans(n_clusters=k)
        self.labels_ = self.model_.fit_predict(X_features)
        self.app_classes_ = self._create_groups(self.app_classes_, self.labels_).reset_index()
        self.app_classes_['app_id'] = self.app_classes_.index
        return self

    def score(self, X:pd.DataFrame, labels:np.array) -> float:
        return davies_bouldin_score(X, labels)

    def analyze_classes(self, df=None, dir=None):
        super().analyze_classes(self.app_classes_, 'app_id', df=df, dir=dir)

    def visualize_tsne(self, df, path=None):
        df = self.standardize(df)
        super().visualize_tsne(df, self.score_names_, sample_size=10000)

    def visualize(self, df, path=None):
        print("Visualizing")
        df = self.standardize(df)
        df.loc[:,self.score_features_] = self.transform_.transform(df[self.score_features_])
        super().visualize(df, self.cluster_features_, path=path)

    def get_magnitude(self, io_identity):
        return super().get_magnitude(io_identity, self.app_classes_)

    def filter_sslos(self, storage_classifier, min_coverage_thresh):
        """
        For each application class, filter out sslos that have little chance of being useful.
        """
        sslos = []
        #Compare every application class with every sslo
        for idx,app_class in self.app_classes_.iterrows():
            coverages = storage_classifier.get_coverages(app_class)
            coverages = coverages[coverages.magnitude >= min_coverage_thresh]
            coverages['app_id'] = app_class['app_id'].astype(int)
            sslos.append(coverages)
        #Add the sslos to the dataframe
        self.app_sslo_mapping = pd.concat(sslos)
        print(self.app_sslo_mapping)

    def get_fitnesses(self, io_identifier:pd.DataFrame) -> pd.DataFrame:
        """
        Determine how well the I/O Identifier fits within each class of behavior
        This assumes the io_identifier has already been standardized
        """
        app_classes = self.app_classes_
        #Get the fitness between io_identifier and every app class (score)
        fitness = 1 - np.abs(app_classes[self.score_names_] - io_identifier[self.score_names_].to_numpy())
        #Add features
        fitness.loc[:,self.features_] = app_classes[self.features_].to_numpy()
        fitness.loc[:,'app_id'] = app_classes['app_id']
        #Get the magnitude of the fitnesses
        fitness.loc[:,"magnitude"] = self.get_magnitude(fitness)
        return fitness
