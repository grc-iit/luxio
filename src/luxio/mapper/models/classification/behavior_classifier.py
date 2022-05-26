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

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
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
        self.score_features_ = None
        self.score_feature_weights_ = None
        self.score_names_ = None
        self.score_weights_ = None
        self.score_weight_sum_ = 0

    def _init_scoring(self):
        # Define the scoring categories
        score_conf = self.score_conf
        # Get score weights and remember the score categories
        self.score_weights_ = []
        for score_name, score_features in score_conf.items():
            score_features = self.score_feature_weights_.columns.intersection(score_features)
            if len(score_features):
                score_weight = self.score_feature_weights_[score_features].to_numpy().sum()
            else:
                score_weight = 0
            self.score_weights_.append(score_weight)
        self.score_weights_ = np.array(self.score_weights_)
        self.score_weight_sum_ = self.score_weights_.sum()
        self.score_weights_ /= self.score_weight_sum_
        self.score_names_ = list(score_conf.keys())

    def standardize(self, data):
        # Define the scoring categories
        score_conf = self.score_conf
        # Get score weights and remember the score categories
        data_trans = pd.DataFrame(self.transform_.transform(data[self.score_features_]), columns=self.score_features_)
        for score_name,score_features in score_conf.items():
            score_features = self.score_feature_weights_.columns.intersection(score_features)
            if len(score_features) == 0:
                data[score_name] = 0
                continue
            score_weights = self.score_feature_weights_[score_features].to_numpy()
            data[score_name] = (data_trans[score_features].to_numpy()*score_weights).sum(axis=1)
        return data

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

    def define_low_med_high(self, dir):
        n = len(self.score_features_)
        scaled_features = pd.DataFrame([[size]*n for size in [.33, .66, 1]], columns=self.score_features_)
        unscaled_features = pd.DataFrame(self.transform_.inverse_transform(scaled_features), columns=self.score_features_)
        unscaled_features[self.score_features_].to_csv(os.path.join(dir, "low_med_high.csv"), index=False)

    def analyze_classes(self, classes, id, df=None, labels=None, dir=None):
        if dir is not None:
            self.define_low_med_high(dir)
            classes = self.standardize(classes.copy())
            #Save each class as a CSV
            if df is not None:
                df = self.standardize(df)
                df['label'] = self.labels_
                for i in range(len(classes)):
                    class_df = df[df.label == i][self.score_features_ + self.score_names_]
                    class_df.loc[:,self.score_features_] = self.transform_.transform(class_df[self.score_features_])
                    class_df.loc[:,self.score_names_] = class_df[self.score_names_].to_numpy() / self.score_weights_
                    class_df.to_csv(os.path.join(dir, f"class_{i}.csv"))
            #Save the median, unweighted class scores
            score_classes = classes[self.score_names_ + ["count", id]]
            score_classes.loc[:,self.score_names_] = score_classes[self.score_names_].to_numpy() / self.score_weights_
            score_classes.sort_values("count", ascending=False, inplace=True)
            score_classes.to_csv(os.path.join(dir, "behavior_score_means.csv"))
            #Save the median stadardized feature values
            feature_classes = classes[self.score_features_ + ["count", id]]
            feature_classes.loc[:,self.score_features_] = (self.transform_.transform(feature_classes[self.score_features_])*3).astype(int)
            for feature in self.score_features_:
                for i,label in enumerate(["low", "medium", "high"]):
                    feature_classes.loc[feature_classes[feature] == i,feature] = label
                feature_classes.loc[feature_classes[feature] == 3,feature] = "high"
            feature_classes.sort_values("count", ascending=False, inplace=True)
            feature_classes.to_csv(os.path.join(dir, "behavior_means.csv"), index=False)

    def visualize(self, df, features, perplexities=None, n_iters=None, learning_rates=None, sample_size=None, path=None):
        if perplexities is None:
            perplexities = [10, 30, 50]
        if n_iters is None:
            n_iters = [1000]
        if learning_rates is None:
            learning_rates = [200]

        if sample_size is not None:
            train_x, test_x, train_y, test_y = train_test_split(df, self.labels_, test_size=sample_size, stratify=self.labels_)
        else:
            test_x = df
            test_y = self.labels_

        X_features = test_x[features]
        for perplexity in perplexities:
            for lr in learning_rates:
                for n_iter in n_iters:
                    print(f"perp={perplexity}, learning_rate={lr}, n_iter={n_iter}")
                    X = TSNE(n_components=2, perplexity=perplexity, learning_rate=lr, n_iter=n_iter, n_jobs=6).fit_transform(X_features)
                    plt.scatter(X[:,0], X[:,1], label=test_y, c=test_y, alpha=.3)
                    plt.show()
                    if path is not None:
                        plt.savefig(path)
                    plt.close()

    @abstractmethod
    def feature_selector(self, X, y):
        return

    @abstractmethod
    def fit(self, X):
        return self

    def get_magnitude(self, data, classes):
        score_names = classes[self.score_names_].columns.intersection(data.columns)
        data = data.fillna(0)
        data[data[score_names] > 1] = 1
        magnitude = (data[score_names].to_numpy() * self.score_weights_)[score_names].to_numpy().sum(axis=1) / self.score_weights_[score_names].sum()
        return magnitude

    def save(self, path):
        pkl.dump(self, open( path, "wb" ))

    @staticmethod
    def load(path):
        return pkl.load(open( path, "rb" ))
