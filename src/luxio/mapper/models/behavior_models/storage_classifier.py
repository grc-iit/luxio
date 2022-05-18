import sys,os
import matplotlib.pyplot as plt

from .behavior_classifier import BehaviorClassifier
from luxio.common.configuration_manager import *
from luxio.mapper.models.common import *

from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.cluster import KMeans

from sklearn.metrics import davies_bouldin_score

import pprint
pp = pprint.PrettyPrinter(depth=6)
pd.options.mode.chained_assignment = None

class StorageClassifier(BehaviorClassifier):
    def __init__(self, features, mandatory_features, output_vars, score_conf, dataset_path, random_seed=132415, n_jobs=4):
        super().__init__(features, mandatory_features, [], score_conf, dataset_path, random_seed, n_jobs)
        self.sslos_ = None #A dataframe containing: means, stds, ns

    def feature_selector(self, X, y):
        self.feature_selector_ = None
        self.feature_importances_ = np.array([1]*len(self.features)) / len(self.features)
        self.features_ = self.features
        self.named_feature_importances_ = pd.DataFrame([(feature, importance) for feature, importance in zip(self.features_, self.feature_importances_)])

    def fit(self, X:pd.DataFrame=None, k=None, score_conf=None):
        # Initialize scoring data
        if score_conf is not None:
            self.score_conf = score_conf
        self.cluster_features_ = list(self.features_)
        self.score_features_ = list(self.features_)
        self.score_default_feature_weights_ = pd.DataFrame(
            dict([(feature, weight) for feature, weight in zip(self.features_, self.feature_importances_)]),
            index=[0]
        )
        self.transform_ = MinMaxScaler().fit(X[self.score_features_])
        self._init_scoring()
        #Cluster data
        self.sslos_ = self.standardize(X)
        X_features = self.sslos_[self.score_features_]
        X_features.loc[:,self.score_features_] = self.transform_.transform(X_features)
        X_features = X_features[self.cluster_features_]
        if k is None:
            for k in [4, 6, 8, 10, 12, 15, 20]:
                self.model_ = KMeans(n_clusters=k)
                self.labels_ = self.model_.fit_predict(X_features)
                print(f"SCORE k={k}: {self.score(X_features, self.labels_)} {self.model_.inertia_}")
            k = int(input("Optimal k: "))
        self.model_ = KMeans(n_clusters=k)
        self.labels_ = self.model_.fit_predict(X_features)
        self.sslos_ = self._create_groups(self.sslos_, self.labels_)
        self.sslos_.rename(columns={"labels":"sslo_id"}, inplace=True)
        self.sslo_to_deployment_ = X
        self.sslo_to_deployment_.loc[:,"sslo_id"] = self.labels_
        return self

    def score(self, X:pd.DataFrame, labels:np.array) -> float:
        return davies_bouldin_score(X, labels)

    def analyze_classes(self, df=None, dir=None):
        super().analyze_classes(self.sslos_, 'sslo_id', df=df, dir=dir)

    def visualize_tsne(self, df, path=None):
        df = self.standardize(df)
        super().visualize_tsne(df, self.cluster_features_, n_iters=[1000], path=path)

    def visualize(self, df, path=None):
        df = self.standardize(df)
        super().visualize_tsne(df, self.cluster_features_, path=path)

    def get_magnitude(self, io_identity):
        return super().get_magnitude(io_identity, self.sslos_)

    def get_coverages(self, io_identifier:pd.DataFrame, sslos:pd.DataFrame=None) -> pd.DataFrame:
        """
        Get the extent to which an sslos is covered by each sslo
        sslos: Either the centroid of an app class or the signature of a unique application
        """
        if sslos is None:
            sslos = self.sslos_
        #Get the coverage between io_identifier and the sslos
        coverage = 1 - (sslos[self.score_names_] - io_identifier[self.score_names_].to_numpy())
        print(coverage)
        print(io_identifier[self.score_names_])
        print(sslos[self.score_names_])
        exit(1)
        #Add features
        coverage.loc[:,self.features_] = sslos[self.features_].to_numpy()
        coverage.loc[:,'sslo_id'] = sslos['sslo_id']
        #Get the magnitude of the fitnesses
        coverage.loc[:,"magnitude"] = self.get_magnitude(coverage)
        return coverage
