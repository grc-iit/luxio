
"""
This file is used for three purposes:
    1) To preprocess a trace dataset
    2) Identify important features relating to I/O performance
    3) To identify classes of I/O behavior based on those features
"""

import sys,os
import pandas as pd
import numpy as np

from parse import *
from clever.dataset import *
from clever.models.regression import *
from clever.transformers import *
from clever.feature_selection import *
from clever.models.clustering import *
from clever.metrics import *

import pprint, warnings
import argparse, configparser

pp = pprint.PrettyPrinter(depth=6)

class ArgumentParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("-t", default=None, help="What your use case is: preprocess, fmodel, fmodel_stats, bmodel, bmodel_stats")
        self.parser.add_argument("-c", default="conf/conf.ini", help="The properties file containing model paths and config parameters. Default: conf/conf.ini")
        self.parser.add_argument("-m", default="Argonne", help="Which data to be preprocessed: Argonne, Tiahu")

    def parse(self):
        args = self.parser.parse_args()
        self.tool = args.t
        self.conf = configparser.ConfigParser()
        self.conf.read(args.c)
        return self

#Feature selection module
def feature_selector(params):
    FEATURES = pd.DataFrame().clever.load_features(params["features"])
    #IMM_FEATURES = pd.DataFrame().clever.load_features(params["imm_features"])
    PERFORMANCE = pd.DataFrame().clever.load_features(params["vars"])
    #Create the training and testing datasets
    df = pd.read_csv(params["trace"])
    train_df,hyper_df,test_df = df.clever.random_sample(.5, .2)
    train_df.to_pickle(params["train"])
    hyper_df.to_pickle(params["hyper"])
    test_df.to_pickle(params["test"])
    #Divide the datasets into x and y
    train_x,train_y = train_df.clever.split(FEATURES,PERFORMANCE)
    hyper_x,hyper_y = hyper_df.clever.split(FEATURES,PERFORMANCE)
    test_x,test_y = test_df.clever.split(FEATURES,PERFORMANCE)
    #Partition by performance
    partitioner = KSegments(k=5).fit(np.array(LogTransformer(base=10,add=1).transform(train_y)).flatten())
    #Select features
    fs = FeatureSelector(
        FEATURES,
        PERFORMANCE,
        XGBRegressor(
            transform_y=LogTransformer(base=10,add=1),
            fitness_metric=PartitionedMetric(partitioner, score=RelativeAccuracyMetric(scale=1, add=1)),
            error_metric=PartitionedMetric(partitioner, score=RMLSEMetric(add=1)))
    )
    fs.select(
        train_x, train_y, hyper_x, hyper_y,
        #search_space=fs.model.get_search_space(),
        max_iter=12,
        max_tunes=0,
        max_tune_iter=0)
    #Save and analyze
    fs.save(params["selector"])
    fs.model.save(params["regressor"])
    fs.save_importances(params["importances"])
    pp.pprint(fs.analyze(test_x,test_y))

#Feature selection statistics module
def feature_selector_stats(params):
    FEATURES = pd.DataFrame().clever.load_features(params["features"])
    PERFORMANCE = pd.DataFrame().clever.load_features(params["vars"])
    #Create the training and testing datasets
    train_df = pd.read_pickle(params["train"])
    hyper_df = pd.read_pickle(params["hyper"])
    test_df = pd.read_pickle(params["test"])
    #Divide the datasets into x and y
    train_x,train_y = train_df.clever.split(FEATURES,PERFORMANCE)
    hyper_x,hyper_y = train_df.clever.split(FEATURES,PERFORMANCE)
    test_x,test_y = train_df.clever.split(FEATURES,PERFORMANCE)
    #Load the feature selector
    fs = FeatureSelector.load(params["selector"])
    pp.pprint(fs.analyze(test_x, test_y))
    fs.save_importances(params["importances"])

#Behavior Classifaction module
def behavior_classifier(params):
    params = args.conf["APP_BEHAVIOR_MODEL"]
    df = pd.read_csv(params["trace"])
    feature_importances = FeatureSelector.load_importances(params["importances"])
    vars = pd.DataFrame().clever.load_features(params["vars"])
    #imm_features = pd.DataFrame().clever.load_features(params["imm_features"])
    reg = FeatureSelector.load(params["regressor"])
    bc = BehaviorClassifier(feature_importances, vars, reg)
    bc.fit(df)
    bc.save(params["classifier"])

#Behavior Classification statistics module
def behavior_classifier_stats(params):
    params = args.conf["APP_BEHAVIOR_MODEL"]
    bc = BehaviorClassifier.load(params["classifier"])
    analysis = bc.analyze()
    pp.pprint(analysis)

##############MAIN##################
if __name__ == "__main__":
    args = ArgumentParser().parse()

    if args.tool == "preprocess":
        params = args.conf["PREPROCESS"]
        parser = ArgonneParser(params["theta_csv"], params["mira_csv"], params["mira_mapping"])
        parser.standardize()
        parser.clean()
        parser.to_csv(params["argonne_csv"])

    if args.tool == "fmodel":
        params = args.conf["APP_BEHAVIOR_MODEL"]
        feature_selector(params)

    if args.tool == "fmodel_stats":
        params = args.conf["APP_BEHAVIOR_MODEL"]
        feature_selector_stats(params)

    if args.tool == "bmodel":
        params = args.conf["APP_BEHAVIOR_MODEL"]
        behavior_classifier(params)

    if args.tool == "bmodel_stats":
        params = args.conf["APP_BEHAVIOR_MODEL"]
        behavior_classifier_stats(params)
