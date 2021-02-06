
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

#Performance variable partitioner
def perf_var_partitioner(params):
    PERFORMANCE = pd.DataFrame().clever.load_features(params["vars"])
    PERF_LABELS = ["label_{}".format(var) for var in PERFORMANCE]
    df = pd.read_csv(params["trace"])
    df.loc[:,PERF_LABELS] = ResolutionReducer(k=4).fit_transform(np.array(LogTransformer(base=10,add=1)(df[PERFORMANCE])))

#Preliminary analysis
def preliminary_analysis(params):
    df = pd.read_csv(params["trace"])
    analysis = pd.DataFrame(df.clever.analyze())
    pp.pprint(analysis)
    df.to_csv(params["trace_analysis"])

#Performance partitioner
def performance_partitioner(params):
    PERFORMANCE = pd.DataFrame().clever.load_features(params["vars"])
    df = pd.read_csv(params["trace"])
    #Partition the different performance variables
    partitioners = KResolutionReducer(k=4).fit(np.array(LogTransformer(base=10,add=1).transform(df[PERFORMANCE])))
    partitioners.save(params["perf_partitions"])
    #Create the training and testing datasets
    train_df,hyper_df,test_df = df.clever.random_sample(.5, .2)
    train_df.to_pickle(params["train"])
    hyper_df.to_pickle(params["hyper"])
    test_df.to_pickle(params["test"])
    return partitioners, train_df, hyper_df, test_df

#Feature selection module
def feature_selector(params):
    FEATURES = pd.DataFrame().clever.load_features(params["features"])
    #IMM_FEATURES = pd.DataFrame().clever.load_features(params["imm_features"])
    PERFORMANCE = pd.DataFrame().clever.load_features(params["vars"])

    #Load the training and testing datasets
    try:
        partitioners = KResolutionReducer.load(params["perf_partitions"])
        train_df = pd.read_pickle(params["train"])
        hyper_df = pd.read_pickle(params["hyper"])
        test_df = pd.read_pickle(params["test"])
    except:
        partitioners, train_df, hyper_df, test_df = performance_partitioner(params)

    #Divide the datasets into x and y
    train_x,train_y = train_df.clever.split(FEATURES,PERFORMANCE)
    hyper_x,hyper_y = hyper_df.clever.split(FEATURES,PERFORMANCE)
    test_x,test_y = test_df.clever.split(FEATURES,PERFORMANCE)

    #Train model for each variable and select minimum feature set
    fs = FeatureSelector(
        FEATURES,
        PERFORMANCE,
        [EnsembleModelRegressor(
            transform_y=LogTransformer(base=10,add=1),
            #fitness_metric=RelativeAccuracyMetric(scale=1, add=1),
            #error_metric=RMLSEMetric(add=1)),
            fitness_metric=PartitionedMetric(partitioner, score=RelativeAccuracyMetric(scale=1, add=1)),
            error_metric=PartitionedMetric(partitioner, score=RMLSEMetric(add=1))
        ) for partitioner in partitioners.segments_]
    )

    fs.select(
        train_x, train_y, hyper_x, hyper_y,
        max_iter=10,
        max_tunes=0,
        max_tune_iter=0,
        growth=.5,
        acc_loss=.05,
        thresh=.02)

    #Save and analyze
    fs.save(params["selector"])
    fs.model_.save(params["regressor"])
    fs.save_importances(params["importances"])
    pp.pprint(fs.analyze(test_x,test_y))

#Feature selection statistics module
def feature_selector_stats(params):
    FEATURES = pd.DataFrame().clever.load_features(params["features"])
    PERFORMANCE = pd.DataFrame().clever.load_features(params["vars"])
    #Create the training and testing datasets
    partitioners = KResolutionReducer.load(params["perf_partitions"])
    train_df = pd.read_pickle(params["train"])
    hyper_df = pd.read_pickle(params["hyper"])
    test_df = pd.read_pickle(params["test"])
    #Divide the datasets into x and y
    train_x,train_y = train_df.clever.split(FEATURES,PERFORMANCE)
    hyper_x,hyper_y = hyper_df.clever.split(FEATURES,PERFORMANCE)
    test_x,test_y = test_df.clever.split(FEATURES,PERFORMANCE)
    #Load the feature selector
    fs = FeatureSelector.load(params["selector"])
    fs.model_._calculate_metrics()
    analysis = fs.analyze(test_x, test_y, metrics=[{
        "r2" : r2Metric(),
        "RelativeError" : PartitionedMetric(partitioner, score=RelativeErrorMetric(add=1), mode='all'),
        "RMLSE" : PartitionedMetric(partitioner, score=RMLSEMetric(add=1), mode='all'),
    } for partitioner in partitioners.segments_])
    pp.pprint(analysis)
    fs.save(params["selector"])
    fs.model_.save(params["regressor"])
    fs.save_importances(params["importances"])

#Behavior Classifaction module
def behavior_classifier(params):
    df = pd.read_csv(params["trace"])
    imm_features = pd.DataFrame().clever.load_features(params["imm_features"])
    vars = pd.DataFrame().clever.load_features(params["vars"])
    fs = FeatureSelector.load(params["selector"])
    bc = BehaviorClassifier(fs.model_.features_, vars, imm_features, fs.model_)
    bc.fit(df)
    bc.save(params["classifier"])

#Behavior Classification statistics module
def behavior_classifier_stats(params):
    #df = pd.read_csv(params["trace"])
    bc = BehaviorClassifier.load(params["classifier"])
    analysis = bc.analyze(dir="datasets/model_analysis")
    #pp.pprint(analysis)
    #bc.visualize()

##############MAIN##################
if __name__ == "__main__":
    args = ArgumentParser().parse()

    if args.tool == "preprocess":
        params = args.conf["PREPROCESS"]
        parser = ArgonneParser(params["theta_csv"], params["mira_csv"], params["mira_mapping"])
        parser.standardize()
        parser.clean()
        parser.to_csv(params["argonne_csv"])

    if args.tool == "preliminary":
        preliminary_analysis(args.conf["PREPROCESS"])

    if args.tool == "dataset_split":
        performance_partitioner(args.conf["APP_BEHAVIOR_MODEL"])

    if args.tool == "fmodel":
        feature_selector(args.conf["APP_BEHAVIOR_MODEL"])

    if args.tool == "fmodel_stats":
        feature_selector_stats(args.conf["APP_BEHAVIOR_MODEL"])

    if args.tool == "bmodel":
        behavior_classifier(args.conf["APP_BEHAVIOR_MODEL"])

    if args.tool == "bmodel_stats":
        behavior_classifier_stats(args.conf["APP_BEHAVIOR_MODEL"])
