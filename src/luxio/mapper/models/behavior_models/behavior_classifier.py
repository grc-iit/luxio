import sys,os
import pickle as pkl
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
import pprint, warnings
pp = pprint.PrettyPrinter(depth=6)

import luxio.common.error_codes
from luxio.common.configuration_manager import *
from luxio.mapper.models.metrics import r2Metric, RelativeErrorMetric, RelativeAccuracyMetric, RMLSEMetric
from luxio.mapper.models.transforms import LogTransformer

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
import umap.plot
from sklearn.model_selection import KFold, train_test_split
import sklearn
import copy

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
        self.score_default_feature_weights_ = None
        self.score_conf_ = None
        self.score_names_ = None
        self.score_weights_ = None
        self.score_weight_sum_ = 0

    def _init_scoring(self):
        # Define the scoring categories
        score_conf = self.score_conf
        # Get score weights and remember the score categories
        self.score_weights_ = []
        self.score_conf_ = {}
        for score_name, score_aggregator in score_conf.items():
            #Decompose the scoring schema
            score_features = score_aggregator['features']
            score_feature_transform = score_aggregator['transform'] if 'transform' in score_aggregator else None
            score_method = score_aggregator['method']
            score_method_type = score_method['type']
            score_weight = score_aggregator['weight'] if 'weight' in score_aggregator else None
            score_features = np.intersect1d(np.array(score_features), np.array(self.features))

            #Initialize score_conf_
            self.score_conf_[score_name] = {
                'features': score_features,
                'transform': None,
                'transform-fitted': False,
                'method': copy.deepcopy(score_method),
                'weight': score_weight
            }

            #Transform the features before aggregation
            if score_feature_transform == 'log10p1':
                self.score_conf_[score_name]['transform'] = LogTransformer(add=1,base=10)
            elif score_feature_transform == 'learned':
                self.score_conf_[score_name]['transform'] = self.transform_
                self.score_conf_[score_name]['transform-fitted'] = True
            else:
                self.score_conf_[score_name]['transform'] = None

            #Combine the features using the aggregation method
            score_feature_weights = None
            if score_method_type == 'weighted-avg':
                score_feature_weights = score_method['weights'] if 'weights' in score_method else None
                if score_feature_weights is None:
                    score_features = self.score_default_feature_weights_.columns.intersection(score_features)
                    score_feature_weights = self.score_default_feature_weights_[score_features].to_numpy()
                    self.score_conf_[score_name]['features'] = score_features
                elif score_feature_weights == 'equal':
                    score_feature_weights = np.array([1/len(score_features)] * len(score_features))
                elif isinstance(score_feature_weights, list):
                    score_feature_weights = np.array(score_feature_weights)
                else:
                    raise Error(ErrorCode.INVALID_FEATURE_WEIGHTS).format(score_feature_weights)
                self.score_conf_[score_name]['method']['weights'] = np.array(score_feature_weights)
            else:
                raise Error(ErrorCode.INVALID_SCORING_METHOD).format(score_method_type)

            #Calculate the weight of the score in factoring fitness
            if score_weight is None:
                if len(score_features) and score_feature_weights is not None:
                    score_weight = score_feature_weights.sum()
                else:
                    score_weight = 0
            self.score_weights_.append(score_weight)

        #Save score weights and names
        self.score_weights_ = np.array(self.score_weights_)
        self.score_weight_sum_ = self.score_weights_.sum()
        self.score_weights_ /= self.score_weight_sum_
        self.score_names_ = list(score_conf.keys())

    def standardize(self, data):
        #Decompose the scoring schema
        for score_name,score_aggregator in self.score_conf_.items():
            score_features = score_aggregator['features']
            score_feature_transform = score_aggregator['transform']
            score_feature_transform_fitted = score_aggregator['transform-fitted']
            score_method = score_aggregator['method']
            score_method_type = score_method['type']
            score_features = np.intersect1d(np.array(data.columns), np.array(score_features))

            #Check if score has any features
            if len(score_features) == 0:
                data[score_name] = 0
                continue

            #Transform the features
            if score_feature_transform is not None:
                if score_feature_transform_fitted:
                    data_trans = pd.DataFrame(score_feature_transform.fit_transform(data[self.score_features_]),
                                              columns=score_features)
                    score_aggregator['transform-fitted'] = True
                else:
                    data_trans = pd.DataFrame(score_feature_transform.transform(data[self.score_features_]),
                                              columns=score_features)
            else:
                data_trans = data[self.score_features_]

            #Combine features
            if score_method_type == 'weighted-avg':
                score_weights = score_method['weights']
                data[score_name] = self._divide_ignore_0(
                    (data_trans[score_features].to_numpy()*score_weights).sum(axis=1),
                    score_weights.sum())
        return data

    def feature_selector_stats(self, X, y, importances_path=None):
        # Divide the datasets into x and y
        train_hyper_x, test_x, train_hyper_y, test_y = train_test_split(X, y, test_size=.1, random_state=self.random_seed)
        # Load the feature selector
        model = self.feature_selector_
        pp.pprint(self.feature_selector_.get_params())
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

    def _divide_ignore_0(self, a, b):
        return np.divide(a, b, out=np.zeros_like(a), where=b != 0)

    def define_low_med_high(self, dir):
        n = len(self.score_features_)
        scaled_features = pd.DataFrame([[size]*n for size in [0, .33, .66, 1]], columns=self.score_features_)
        unscaled_features = pd.DataFrame(self.transform_.inverse_transform(scaled_features), columns=self.score_features_)
        unscaled_features.to_csv(os.path.join(dir, "low_med_high_0_33_66_1.csv"), index=False)

        scaled_features = pd.DataFrame([[size] * n for size in [0, .25, .5, .75, 1]], columns=self.score_features_)
        unscaled_features = pd.DataFrame(self.transform_.inverse_transform(scaled_features), columns=self.score_features_)
        unscaled_features.to_csv(os.path.join(dir, "low_med_high_0_25_50_75_1.csv"), index=False)

    def analyze_classes(self, classes, id, df=None, labels=None, dir=None):
        self.define_low_med_high(dir)
        if dir is not None:
            classes = self.standardize(classes.copy())
            #Save each class as a CSV
            if df is not None:
                df = self.standardize(df)
                df['label'] = self.labels_
                df.to_csv(os.path.join(dir, f"unscaled_class_all.csv"))
                #complete, transformed
                df.loc[:, self.score_features_] = self.transform_.transform(df[self.score_features_])
                df.to_csv(os.path.join(dir, f"scaled_class_all.csv"))
                #feature reduced, transformed
                df[self.score_features_ + ['label']].to_csv(os.path.join(dir, f"scaled_class_subset.csv"))

            #Save the median, unweighted class scores
            score_classes = classes[self.score_names_ + ["count", id]]
            score_classes.loc[:,self.score_names_] = score_classes[self.score_names_].to_numpy()
            score_classes.sort_values("count", ascending=False, inplace=True)
            score_classes.to_csv(os.path.join(dir, "behavior_score_means.csv"))
            #Save the median stadardized feature values
            feature_classes = classes[self.score_features_ + ["count", id]]
            feature_classes.loc[:,self.score_features_] = self.transform_.transform(feature_classes[self.score_features_])
            feature_classes.to_csv(os.path.join(dir, "behavior_means_all.csv"), index=False)
            #Save the median semantic feature values
            feature_classes.loc[:,self.score_features_] = (feature_classes[self.score_features_]*(1/.3)).astype(int)
            for feature in self.score_features_:
                for i,label in enumerate(["low", "medium", "high"]):
                    feature_classes.loc[feature_classes[feature] == i,feature] = label
                feature_classes.loc[feature_classes[feature] == 3,feature] = "high"
            feature_classes.sort_values("count", ascending=False, inplace=True)
            feature_classes.to_csv(os.path.join(dir, "behavior_means_all_semantic.csv"), index=False)
            feature_classes[self.cluster_features_ + ["count", id]].to_csv(os.path.join(dir, "behavior_means_subset_semanitc.csv"), index=False)

    def visualize_tsne(self, df, features, perplexities=None, n_iters=None, learning_rates=None, sample_size=None, path=None):
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

    def visualize(self, df, features, path=None):
        print("FITTING")
        mapper = umap.UMAP().fit(df[features])
        print("DRAWING")
        p = umap.plot.points(mapper, labels=self.labels_, color_key_cmap='tab10')
        umap.plot.show(p)

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
        magnitude = (data[score_names].to_numpy() * self.score_weights_).sum(axis=1)
        return magnitude

    def save(self, path):
        pkl.dump(self, open( path, "wb" ))

    @staticmethod
    def load(path):
        return pkl.load(open( path, "rb" ))
