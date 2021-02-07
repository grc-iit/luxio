import sys,os
from .generic_cluster import GenericCluster
from sklearn.manifold import TSNE
import numpy as np
from matplotlib import pyplot as plt

from sklearn.cluster import DBSCAN, KMeans
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import FunctionTransformer, StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score

from .kresolution_reducer import KResolutionReducer
from clever.dataset import *
from clever.transformers import *
from clever.metrics import *
import numpy as np
import pandas as pd

import pickle as pkl

import pprint, warnings
pp = pprint.PrettyPrinter(depth=6)

class BehaviorClassifier(GenericCluster):
    def __init__(self, features, vars, imm_features, regressor):
        self.features = features
        self.feature_importances = np.array(regressor.feature_importances_)
        self.vars = list(vars)
        self.imm_features = imm_features
        self.regressor = regressor
    
    def fit(self, X):
        return

    def predict(self, X):
        return

    def predict_proba(self, X):
        return
