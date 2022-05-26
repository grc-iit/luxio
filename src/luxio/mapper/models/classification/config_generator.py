import sys,os
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np

from .behavior_classifier import BehaviorClassifier
from luxio.common.configuration_manager import *
from luxio.mapper.models.common import *
from luxio.mapper.models.regression.forest.random_forest import RFERandomForestRegressor
from luxio.mapper.models.metrics import r2Metric, RelativeAccuracyMetric, RelativeErrorMetric
from luxio.mapper.models.transforms.transformer_factory import TransformerFactory
from luxio.mapper.models.transforms.log_transformer import LogTransformer
from luxio.mapper.models.transforms.chain_transformer import ChainTransformer
from luxio.mapper.models.dimensionality_reduction.dimension_reducer_factory import DimensionReducerFactory,RandomForestReducer

from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.metrics import davies_bouldin_score, r2_score
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

import itertools
import pickle as pkl

import pprint, warnings
pp = pprint.PrettyPrinter(depth=6)
pd.options.mode.chained_assignment = None

class ConfigGenerator:
    def __init__(self,
                 stress_test_config_features, stress_test_dpl_features, stress_test_output_vars,
                 sslo_features, sslo_output_vars, random_seed=132415, n_jobs=6):
        self.stress_test_config_features = stress_test_config_features
        self.stress_test_dpl_features = stress_test_dpl_features
        self.stress_test_output_vars = stress_test_output_vars
        self.sslo_features = sslo_features
        self.sslo_output_vars = sslo_output_vars
        self.random_seed = random_seed
        self.n_jobs = n_jobs

    def _normalize_grps(self, df, grp_features):
        """
        grps = df.groupby(grp_features)
        for grp_name,grp_df in grps:
            df = grp_df.reset_index()
            if len(df) > 20:
                return df
        """

        grps = df.groupby(grp_features)
        df.loc[:, self.stress_test_output_vars] = grps.transform(lambda x: (x - x.mean()) / x.std()).reset_index()
        df = df.fillna(0)

        """
        sub_df = df[self.stress_test_output_vars]
        sub_df[sub_df < 0] = -1
        sub_df[sub_df > 0] = 1
        df.loc[:,self.stress_test_output_vars] = sub_df
        """

        return df

    #Get which configuration parameters impact performance
    #Group by device_type, network, num_servers, io_type, req_size
    #Normalize each group
    #Predict normalized read/write BW
    def config_feature_select(self, df):
        df = self._normalize_grps(df, self.stress_test_dpl_features)
        features = self.stress_test_config_features + self.stress_test_dpl_features
        X = df[features]
        y = df[self.stress_test_output_vars]
        param_grid = {
            'n_features': [8],
            'max_depth': [2, 5, 8, 15, 50],
            'n_estimators': [5, 10]
        }
        print("Fitting Model")

        model = RandomForestReducer(features, max_depth=9)
        model.fit(X,y)
        self.feature_selector_ = model
        self.named_feature_importances_ = model.sorted_features_

        """
        model = RFERandomForestRegressor(
            features=features,
            fitness_metric=r2Metric(),
            error_metric=RelativeErrorMetric())
        search = GridSearchCV(model, param_grid, cv=KFold(n_splits=20, random_state=self.random_seed, shuffle=True),
                              n_jobs=self.n_jobs, verbose=2)
        search.fit(X, y)
        self.feature_selector_ = search.best_estimator_
        self.feature_importances_ = self.feature_selector_.feature_importances_
        self.features_ = self.feature_selector_.features_
        self.named_feature_importances_ = pd.DataFrame(
            [(feature, importance) for feature, importance in zip(self.features_, self.feature_importances_)])
        """

    def config_feature_select_stats(self, df):
        """
        df = self._normalize_grps(df, self.stress_test_dpl_features)
        features = self.stress_test_config_features + self.stress_test_dpl_features
        pred_y = self.feature_selector_.predict(df[features])
        y = df[self.stress_test_output_vars]
        print(pd.DataFrame(pred_y, columns=self.stress_test_output_vars))
        print(y)
        """

        print(f"Model Score: {self.feature_selector_.fitness_}")
        #print(f"Overall r2 Score: {r2_score(pred_y, df[self.stress_test_output_vars])}")
        pp.pprint(self.named_feature_importances_)

    #Given device_type, network, num_devices, config, predict bw set
    def fit(self, X, y):
        train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=.1, random_state=self.random_seed)
        param_grid = {
            'n_features': [15, 20, 25],
            'max_depth': [5, 10],
            'reducer_method': ['random-forest']
        }
        print("Heuristic Feature Reduction")
        heuristic_reducer = DimensionReducerFactory(features=self.features, n_jobs=self.n_jobs)
        heuristic_reducer.fit(X,y)
        print("Fitting Feature Reduced Model")
        model = RFERandomForestRegressor(
            features=self.features,
            transform_y=LogTransformer(add=1,base=10),
            heuristic_reducer=heuristic_reducer,
            n_features_heur=35,
            fitness_metric=RelativeAccuracyMetric(),
            error_metric=RelativeErrorMetric())
        search = GridSearchCV(model, param_grid, cv=KFold(n_splits=5, random_state=self.random_seed, shuffle=True), n_jobs=self.n_jobs, verbose=2)
        search.fit(train_x, train_y)
        self.model_ = search.best_estimator_
        self.feature_importances_ = self.feature_selector_.feature_importances_
        self.features_ = self.feature_selector_.features_
        self.named_feature_importances_ = pd.DataFrame([(feature,importance) for feature,importance in zip(self.features_, self.feature_importances_)])

    #Given device_type, network, max num_devices, target_bws, get config
    def get_config(self, X, max_devs, target_bws):
        return

    def save(self, path):
        pkl.dump(self, open( path, "wb" ))

    @staticmethod
    def load(path):
        return pkl.load(open( path, "rb" ))


