
"""
This file is used for three purposes:
    1) To preprocess a trace dataset
    2) Identify important features relating to I/O performance
    3) To identify classes of I/O behavior based on those features
"""

import sys,os
import pandas as pd
import numpy as np

from clever.dataset import *
from clever.models.regression import *
from clever.transformers import *
from clever.feature_selection import *
from clever.models.clustering import *
from clever.metrics import *
from sklearn.preprocessing import MinMaxScaler

import pprint, warnings
import argparse, configparser

pp = pprint.PrettyPrinter(depth=6)

class ArgumentParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("-t", default=None, help="What your use case is: fmodel, fmodel_stats, bmodel, bmodel_stats")
        self.parser.add_argument("-c", default="conf/conf.ini", help="The properties file containing model paths and config parameters. Default: conf/conf.ini")

    def parse(self):
        args = self.parser.parse_args()
        self.tool = args.t
        self.conf = configparser.ConfigParser()
        self.conf.read(args.c)
        return self

##############MAIN##################
if __name__ == "__main__":
    args = ArgumentParser().parse()

    if args.tool == "preprocess":
        a = 0

    if args.tool == "fmodel":
        params = args.conf["APP_BEHAVIOR_MODEL"]
        FEATURES = pd.DataFrame().clever.load_features(params["features"])
        PERFORMANCE = pd.DataFrame().clever.load_features(params["vars"])
        #Create the training and testing datasets
        df = pd.read_csv(params["trace"])
        train_df,hyper_df,test_df = df.clever.random_sample(.5, .2)
        train_x,train_y = train_df.clever.split(FEATURES,PERFORMANCE)
        hyper_x,hyper_y = train_df.clever.split(FEATURES,PERFORMANCE)
        #Select features
        fs = FeatureSelector(
            FEATURES,
            PERFORMANCE,
            PartitionedEnsembleRegressor(
                transform_y=LogTransformer(base=10,add=1),
                fitness_metric=RelativeAccuracyMetric(add=1),
                error_metric=RMLSEMetric(add=1))
        )
        fs.select(train_x, train_y, hyper_x, hyper_y, fs.model.tune_config(train_x, train_y, hyper_x, hyper_y), max_tune_iter=0)
        #Save and analyze
        fs.save(params["selector"])
        fs.model.save(params["regressor"])
        fs.save_importances(params["importances"])
        fs.analyze()

    if args.tool == "fmodel_stats":
        params = args.conf["APP_BEHAVIOR_MODEL"]
        fs = FeatureSelector.load(params["selector"])
        fs.analyze()
        fs.save_importances(params["importances"])

    if args.tool == "bmodel":
        params = args.conf["APP_BEHAVIOR_MODEL"]
        df = pd.read_csv(params["trace"])
        importances = FeatureSelector.load_importances(params["importances"])
        vars = pd.DataFrame().clever.load_features(params["vars"])
        reg = FeatureSelector.load(params["regressor"])
        wic = BehaviorClassifier(importances, vars, reg)
        wic.fit(df)
        wic.save(params["classifier"])

    if args.tool == "bmodel_stats":
        params = args.conf["APP_BEHAVIOR_MODEL"]
        wic = BehaviorClassifier.load(params["classifier"])
        analysis = wic.log_clusters.analyze(method='stat-clust', metric='mean')
        analysis_df = pd.DataFrame(analysis).clever.transform(MinMaxScaler(), fit=True, features=wic.features + wic.vars)
        analysis_df.transpose().to_csv(params["class_metrics"])
