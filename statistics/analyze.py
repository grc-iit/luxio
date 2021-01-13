
"""
This file is used for three purposes:
    1) To preprocess a trace dataset
    2) Identify important features relating to I/O performance
    3) To identify classes of I/O behavior based on those features
"""

import sys,os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from clever.dataset import *
from clever.models.regression import *
from clever.transformers import *
from clever.feature_selection import *
from clever.models.clustering import *

import pprint, warnings
import argparse

pp = pprint.PrettyPrinter(depth=6)

class ArgumentParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("-tool", default=None, help="What your use case is: partition, fmodel, fmodel_stats, bmodel, bmodel_stats")
        self.parser.add_argument("-trace", default="datasets/preprocessed_dataset.csv", help="The path to the trace data")
        self.parser.add_argument("-m", default="datasets/model/model.pkl", help="The file to save the feature reduction or app behavior model to")
        self.parser.add_argument("-f", default="features/features.csv", help="The CSV containing features")
        self.parser.add_argument("-v", default="features/performance.csv", help="The CSV containing variables")
        self.parser.add_argument("-i", default="features/importances.csv", help="The CSV containing feature importances")

    def parse(self):
        args = self.parser.parse_args()
        self.tool = args.tool
        self.trace_path = args.trace
        self.model_path = args.m
        self.features_path = args.f
        self.vars_path = args.v
        self.importances_path = args.i
        return self

##############MAIN##################
if __name__ == "__main__":
    args = ArgumentParser().parse()

    if args.tool == "partition":
        lt = LogTransformer(base=10, add = 1)
        df = pd.read_csv(args.trace_path)
        var = pd.read_csv(args.vars_path)
        df.loc[:,var] = df.clever.transform(lt, True)
        partitions = df.clever.kmeans(features=var, k=10, cluster_col="partition").agglomerate(dist_thresh=.5)
        #df_wrap = df.clever.inverse(lt)
        #partitions = df.clever.exp_partition(var, base=10, exp=2.7, min_n=500, cluster_col="partition")
        pp.pprint(partitions.analyze()[var])
        #pd.DataFrame(analysis["TOTAL_IO_TIME"]).transpose().round(decimals=3).to_csv("datasets/partition-stats.csv")

    if args.tool == "fmodel":
        FEATURES = pd.DataFrame().clever.load_features(args.features_path)
        PERFORMANCE = pd.DataFrame().clever.load_features(args.vars_path)
        fs = FeatureSelector(FEATURES, PERFORMANCE, args.trace_path, model_path=args.model_path)
        fs.create_model()
        fs.analyze_model()
        fs.save_importances(args.importances_path)

    if args.tool == "fmodel_stats":
        fs = FeatureSelector.load(args.model_path)
        fs.analyze_model()
        fs.save_importances(args.importances_path)

    if args.tool == "bmodel":
        df = pd.read_csv(args.trace_path)
        importances = FeatureSelector.load_importances(args.importances_path)
        wic = WeightedImportanceClassifier(importances)
        wic.fit(df)
        wic.save(args.model_path)

    if args.tool == "bmodel_stats":
        FEATURES = []
        PERFORMANCE = []
        if args.features_path != None:
            FEATURES = pd.DataFrame().clever.load_features(args.features_path)
        elif args.vars_path != None:
            PERFROMANCE = pd.DataFrame().clever.load_features(args.vars_path)
        wic.model.analyze(FEATURES + PERFROMANCE)
