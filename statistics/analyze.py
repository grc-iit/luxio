
"""
This file is used for three purposes:
    1) Identify important features relating to I/O and to identify classes of I/O
behavior.

For feature reduction, the output of this file will be a csv containing feature names and importances.
"""

import sys,os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from clever.dataset import *
from clever.models import *
from clever.transformers import *
from clever.feature_selection import *

import pprint, warnings

def usage():
    print("Usage: ")

##############MAIN##################
if __name__ == "__main__":
    if case == 1:
        lt = LogTransformer(base=10, add = 1)
        df = pd.read_csv("datasets/preprocessed_dataset.csv")
        #df = df.clever.transform(lt, True)
        #partitions = df.clever.kmeans(features="TOTAL_IO_TIME", k=10, cluster_col="partition").agglomerate(dist_thresh=.5)
        #df_wrap = df.clever.inverse(lt)
        partitions = df.clever.exp_partition("TOTAL_IO_TIME", base=10, exp=2.7, min_n=500, cluster_col="partition")
        pp.pprint(partitions.analyze()["TOTAL_IO_TIME"])
        #pd.DataFrame(analysis["TOTAL_IO_TIME"]).transpose().round(decimals=3).to_csv("datasets/partition-stats.csv")

    if case == 2:
        FEATURES = pd.DataFrame().clever.load_features("features/features.csv")
        PERFORMANCE = pd.DataFrame().clever.load_features("features/performance.csv")
        fs = FeatureSelector(FEATURES, PERFORMANCE, "datasets/preprocessed_dataset.csv", model_dir="datasets/model")
        fs.create_model()

    if case == 3:
        fs = FeatureSelector.load("datasets/model/model.pkl")
        fs.analyze_model()
