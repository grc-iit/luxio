import sys,os
import pickle as pkl
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
import pprint, warnings
pp = pprint.PrettyPrinter(depth=6)

from luxio.common.configuration_manager import *
from luxio.mapper.models.metrics import r2Metric, RelativeErrorMetric, RelativeAccuracyMetric, RMLSEMetric

from sklearn.model_selection import KFold, train_test_split

class BehaviorClassifier(ABC):
    def __init__(self, features, mandatory_features, output_vars, score_conf, dataset_path, random_seed, n_jobs):
        self.features = features
        self.mandatory_features = mandatory_features
        self.output_vars = output_vars
        self.score_conf = score_conf
        self.dataset_path = dataset_path
        self.random_seed = random_seed
        self.n_jobs = n_jobs

        self.feature_selector_ = None
        self.feature_importances_ = None
        self.model_ = None

    def feature_selector_stats(self, X, y, importances_path=None):
        # Divide the datasets into x and y
        train_hyper_x, test_x, train_hyper_y, test_y = train_test_split(X, y, test_size=.1, random_state=self.random_seed)
        # Load the feature selector
        model = self.feature_selector_
        # Determine how well model performs per-features
        analysis = model.analyze(test_x, test_y, metrics={
            "r2Metric":     r2Metric(mode='all'),
            "MAPE-AVG":     RelativeErrorMetric(add=1),
            "RMLSE-AVG":    RMLSEMetric(add=1),
            "MAPE-ALL":     RelativeErrorMetric(add=1, mode='all'),
            "RMLSE-ALL":    RMLSEMetric(add=1, mode='all')
        })
        pp.pprint(analysis)
        if importances_path is not None:
            model.save_importances(importances_path)

    def _smash(self, df:pd.DataFrame, cols:np.array):
        grp = df.groupby(cols)
        medians = grp.median().reset_index()
        #std_col_map = {orig_col:f"std_{orig_col}" for orig_col in means.columns}
        #std_cols = list(std_col_map.values())
        #stds = grp.std().reset_index().rename(std_col_map)
        ns = grp.size().reset_index(name="count")["count"].to_numpy()/len(df)
        idxs = np.argsort(-ns)
        medians = medians.iloc[idxs,:]
        #stds = stds.iloc[idxs,:]
        ns = ns[idxs]
        #means.loc[:,std_cols] = stds.to_numpy()
        medians.loc[:,"count"] = ns
        return medians

    def _create_groups(self, df:pd.DataFrame, labels:np.array):
        df = pd.DataFrame(df)
        df.loc[:,"labels"] = labels
        return self._smash(df, ["labels"])

    @abstractmethod
    def feature_selector(self, X, y):
        return

    @abstractmethod
    def fit(self, X):
        return self

    @abstractmethod
    def analyze_classes(self, dir=None):
        return

    @abstractmethod
    def get_magnitude(self, X):
        return

    @abstractmethod
    def standardize(self, features:pd.DataFrame):
        return

    def save(self, path):
        pkl.dump(self, open( path, "wb" ))

    @staticmethod
    def load(path):
        return pkl.load(open( path, "rb" ))
