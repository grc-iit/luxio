import sys,os
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

from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.cluster import KMeans

from sklearn.metrics import davies_bouldin_score

import pprint, warnings
pp = pprint.PrettyPrinter(depth=6)
pd.options.mode.chained_assignment = None

class StorageClassifier(BehaviorClassifier):
    def __init__(self, features, mandatory_features, output_vars, score_conf, dataset_path, random_seed=132415, n_jobs=4):
        super().__init__(features, mandatory_features, output_vars, score_conf, dataset_path, random_seed, n_jobs)
        self.sslos = None #A dataframe containing: means, stds, ns

    def feature_selector(self, X, y):
        train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=.1, random_state=self.random_seed)
        param_grid = {
            'n_features': [5, 10, 14],
            'max_depth': [5, 10],
            'reducer_method': ['random-forest']
        }
        print("Heuristic Feature Reduction")
        heuristic_reducer = DimensionReducerFactory(features=self.features, n_jobs=self.n_jobs)
        heuristic_reducer.fit(X, y)
        print("Fitting Feature Reduced Model")
        model = RFERandomForestRegressor(
            features=self.features,
            # transform=TransformerFactory(),
            transform_y=LogTransformer(add=1, base=10),
            heuristic_reducer=heuristic_reducer,
            n_features_heur=35,
            fitness_metric=RelativeAccuracyMetric(),
            error_metric=RelativeErrorMetric())
        search = GridSearchCV(model, param_grid, cv=KFold(n_splits=5, random_state=self.random_seed, shuffle=True),
                              n_jobs=self.n_jobs, verbose=2)
        search.fit(train_x, train_y)
        self.feature_selector_ = search.best_estimator_
        self.feature_importances_ = self.feature_selector_.feature_importances_
        self.features_ = self.feature_selector_.features_
        self.named_feature_importances_ = pd.DataFrame([(feature, importance) for feature, importance in zip(self.features_, self.feature_importances_)])

    def fit(self, X:pd.DataFrame=None, k=None):
        X = X.drop_duplicates()
        #Identify clusters of transformed data
        self.transform_ = MinMaxScaler()
        weights = np.array(list(self.feature_importances_) + [1/len(self.output_vars)]*len(self.output_vars))
        X_features = self.transform_.fit_transform(X[self.features_ + self.output_vars])*weights
        if k is None:
            for k in [4, 6, 8, 10, 12, 15, 20]:
                self.model_ = KMeans(n_clusters=k)
                self.labels_ = self.model_.fit_predict(X_features)
                print(f"SCORE k={k}: {self.score(X_features, self.labels_)} {self.model_.inertia_}")
            k = int(input("Optimal k: "))
        self.model_ = KMeans(n_clusters=k)
        self.labels_ = self.model_.fit_predict(X_features)
        #Cluster non-transformed data
        self.sslos = self.standardize(X)
        self.sslos = self._create_groups(self.sslos, self.labels_)
        self.sslos.rename(columns={"labels":"sslo_id"}, inplace=True)
        self.sslo_to_deployment = X
        self.sslo_to_deployment.loc[:,"sslo_id"] = self.labels_
        return self

    def score(self, X:pd.DataFrame, labels:np.array) -> float:
        return davies_bouldin_score(X, labels)

    def define_low_med_high(self, dir):
        SCORES = self.score_conf
        n = len(self.features)
        scaled_features = pd.DataFrame([[size]*n for size in [.33, .66, 1]], columns=self.features)
        unscaled_features = pd.DataFrame(self.transform_.inverse_transform(scaled_features), columns=self.features)
        for score_name,features,score_weight in zip(SCORES.keys(), SCORES.values(), self.score_weights):
            features = self.feature_importances_.columns.intersection(features)
            unscaled_features.loc[:,score_name] = (unscaled_features[features] * self.feature_importances_[features].to_numpy()).sum(axis=1).to_numpy()/score_weight
        unscaled_features[self.scores].to_csv(os.path.join(dir, "low_med_high.csv"))

    def analyze_classes(self, dir=None):
        if dir is not None:
            #trans = ChainTransformer([LogTransformer(base=10,add=1), MinMaxScaler()]).fit(self.sslo_to_deployment)
            self.define_low_med_high(dir)
            sslos = self.sslos.copy()
            #Apply standardization
            sslos[self.scores].to_csv(os.path.join(dir, "orig_behavior_means.csv"))
            #Apply transformation to features
            sslos.loc[:,self.features] = (self.transform_.transform(sslos[self.features])*3).astype(int)
            #Apply transformation to scores
            sslos.loc[:,self.scores] = (sslos[self.scores]*3).fillna(0).astype(int)
            #Label each bin
            for feature in self.features + self.scores:
                for i,label in enumerate(["low", "medium", "high"]):
                    sslos.loc[sslos[feature] == i,feature] = label
                sslos.loc[sslos[feature] == 3,feature] = "high"
            #Store the application classes
            sslos = sslos[self.scores + self.features + ["count"]]
            sslos = sslos.groupby(self.scores).sum().reset_index()
            sslos.sort_values("count", ascending=False, inplace=True)
            sslos[self.scores + ["count"]].to_csv(os.path.join(dir, "behavior_means.csv"))

    def visualize(self, df, path=None):
        df = self.standardize(df)
        weights = np.array(list(self.feature_importances_) + [1 / len(self.output_vars)] * len(self.output_vars))
        X_features = self.transform_.fit_transform(df[self.features_ + self.output_vars]) * weights
        for lr in [200]:
            for perplexity in [2, 5, 10, 20, 30, 50]:
                print(f"PERPLEXITY: {perplexity}")
                X = TSNE(n_components=2, perplexity=perplexity, learning_rate=lr, n_jobs=6).fit_transform(X_features)
                plt.scatter(X[:,0], X[:,1], label=self.labels_, c=self.labels_, alpha=.3)
                plt.show()
                if path is not None:
                    plt.savefig(path)
                plt.close()

    def standardize(self, sslos:pd.DataFrame):
        return sslos
        SCORES = self.score_conf
        #Get score weights and remember the score categories
        if self.scores is None:
            self.scores = list(SCORES.keys())
            self.score_weights = []
            for features in SCORES.values():
                features = self.feature_importances_.columns.intersection(features)
                self.score_weights.append(self.feature_importances_[features].to_numpy().sum())
            self.score_weights = pd.Series(self.score_weights, index=self.scores) / np.sum(self.score_weights)

        scaled_features = pd.DataFrame(self.transform_.transform(sslos[self.features].astype(float)), columns=self.features)
        for score_name,features,score_weight in zip(SCORES.keys(), SCORES.values(), self.score_weights):
            features = scaled_features.columns.intersection(features)
            sslos.loc[:,score_name] = (scaled_features[features] * self.feature_importances_[features].to_numpy()).sum(axis=1).to_numpy()/score_weight

        return sslos

    def get_magnitude(self, coverage:pd.DataFrame):
        scores = self.sslos[self.scores].columns.intersection(coverage.columns)
        coverage = coverage.fillna(0)
        coverage[coverage[scores] > 1] = 1
        return ((coverage[scores]*self.score_weights).sum(axis=1)/np.sum(self.score_weights)).to_numpy()

    def get_coverages(self, io_identifier:pd.DataFrame, sslos:pd.DataFrame=None) -> pd.DataFrame:
        """
        Get the extent to which an sslos is covered by each sslo
        sslos: Either the centroid of an app class or the signature of a unique application
        """
        if sslos is None:
            sslos = self.sslos
        #Get the coverage between sslos and every sslo
        coverage = 1 - (sslos[self.scores] - io_identifier[self.scores].to_numpy())
        #Add features
        coverage.loc[:,self.features] = sslos[self.features].to_numpy()
        coverage.loc[:,'sslo_id'] = sslos['sslo_id']
        #Get the magnitude of the fitnesses
        coverage.loc[:,"magnitude"] = self.get_magnitude(coverage)
        return coverage
