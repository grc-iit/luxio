#SPSS: Logistic Regression
#Stratified random sample
#Bagging and feature reduction
#Look at beacon traces and derive more dataset

#What is performance?
#How much does IO_TIME contribute to RUN_TIME?
    #Data vs compute intensive?
    #Smaller runtime, better performance?
#Predict vector of times instead of just RUN_TIME?

#We need a measurement of accuracy that isn't biased towards large values
#How do we combine models feature importances?
#Partition dataset
#Research stats papers to see how much detail they have

import sys,os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from clever.dataset import *
from clever.models import *
from clever.transformers import *

from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from mord import OrdinalRidge

from sklearn.preprocessing import FunctionTransformer

import pprint, warnings

warnings.simplefilter("ignore") #IGNORE invalid pandas warnings
pp = pprint.PrettyPrinter(depth=6)
SEED = 123

##############HELPER FUNCTIONS##############

def partition_dataset(DATASET, case):
    partitions = None
    if case == 1:
        #partitions = Dataset.read_csv(DATASET).partition("TOTAL_IO_TIME", step=.1, scale=10, min_range=100, cluster_col="partition")
        df = Dataset.read_csv(DATASET)
        df.partition("TOTAL_IO_TIME", step=.1, scale=10, min_range=100, cluster_col="partition")
        partitions = df.divide(cluster_col="partition")
        partitions.save("datasets/partitioned_dataset.pkl")
    elif case == 2:
        partitions = Clusters.load("datasets/partitioned_dataset.pkl")
    return partitions

def sample_dataset(partitions,case):
    if case <= 2:
        train_df,hyper_df,test_df = partitions.random_sample(.5)
        train_df.save("datasets/train.pkl")
        #hyper_df.to_csv("datasets/hyper_{}.pkl")
        test_df.save("datasets/test.pkl")
    else:
        train_df = Clusters.load("datasets/train.pkl")
        #hyper_df = Clusters.read_csv("datasets/hyper.pkl")
        test_df = Clusters.load("datasets/test.pkl")
    return train_df,test_df

def stacked_model_per_partition(case):
    #STEP 1: Partition the dataset using TOTAL_IO_TIME
    print("PARTITIONING DATASET")
    partitions = partition_dataset("datasets/preprocessed_dataset.csv", case)

    #Load features and performance
    FEATURES = list(pd.read_csv("features/features.csv", header=None).iloc[:,0])
    PERFORMANCE = list(pd.read_csv("features/performance.csv", header=None).iloc[:,0])

    #Sample from each partition
    print("GATHERING PER-PARTITION SAMPLES")
    train_dfs,test_dfs = sample_dataset(partitions,case)
    train_x_dfs,train_y_dfs = train_dfs.split(FEATURES,PERFORMANCE)
    test_x_dfs,test_y_dfs = test_dfs.split(FEATURES,PERFORMANCE)

    #Create an ensemble model per-partition
    print("ENSEMBLING")
    models = [
        ("RandomForestRegressor", ForestWrapper(RandomForestRegressor(n_estimators=3, max_leaf_nodes=256, random_state=1, verbose=0))),
        ("XGBRegressor", ForestWrapper(XGBRegressor(objective ='reg:squarederror', n_estimators = 1, seed = 123, verbosity=0))),
        ("AdaBoostRegressor", ForestWrapper(AdaBoostRegressor(loss ='linear', n_estimators = 6)))#,
        #("LinearRegression", CurveWrapper(LinearRegression(fit_intercept=False))),
    ]
    ensemble = EnsembleModelRegressor(
        models,
        combiner_model=ForestWrapper(RandomForestRegressor(n_estimators=5, max_leaf_nodes=256, random_state=1, verbose=0))
    )
    model = PartitionedModelRegressor(
        ensemble,
        ForestWrapper(RandomForestRegressor(n_estimators=5, max_depth=6, random_state=1, verbose=0)),
        transform_y = LogTransformer(base=10,add=1)
    )
    model.fit(train_x_dfs, train_y_dfs)
    model.save("datasets/model.pkl")
    print(model.fitnesses_)
    print(model.fitness_)

def analyze_stacked_model():
    #Load dataset
    print("GATHERING PER-PARTITION SAMPLES")
    FEATURES = list(pd.read_csv("features/features.csv", header=None).iloc[:,0])
    PERFORMANCE = list(pd.read_csv("features/performance.csv", header=None).iloc[:,0])
    train_dfs,test_dfs = sample_dataset(None,3)
    train_x_dfs,train_y_dfs = train_dfs.split(FEATURES,PERFORMANCE)
    test_x_dfs,test_y_dfs = test_dfs.split(FEATURES,PERFORMANCE)

    #Create an ensemble model per-partition
    print("ENSEMBLING")
    model = PartitionedModelClassifier.load("datasets/model.pkl")
    print(model.rmses_)

def ensemble_log_regression(case):
    #STEP 1: Partition the dataset using TOTAL_IO_TIME
    print("PARTITIONING DATASET")
    if case == 1:
        df = Dataset.read_csv("datasets/preprocessed_dataset.csv")

    #Load features and performance
    FEATURES = list(pd.read_csv("features/features.csv", header=None).iloc[:,0])
    PERFORMANCE = list(pd.read_csv("features/performance.csv", header=None).iloc[:,0])

    #Sample from each partition
    print("GATHERING PER-PARTITION SAMPLES")
    if case <= 2:
        train_df,hyper_df,test_df = df.random_sample(.5)
        train_df.save("datasets/train_whole.pkl")
        test_df.save("datasets/test_whole.pkl")
    else:
        train_df = Dataset.load("datasets/train_whole.pkl")
        test_df = Dataset.load("datasets/test_whole.pkl")
    train_x_df,train_y_df = train_df.split(FEATURES,PERFORMANCE)
    test_x_df,test_y_df = test_df.split(FEATURES,PERFORMANCE)

    #Create an ensemble model per-partition
    print("ENSEMBLING")
    models = [
        ("RandomForestRegressor", ForestWrapper(RandomForestRegressor(n_estimators=3, max_leaf_nodes=256, random_state=1, verbose=0))),
        ("XGBRegressor", ForestWrapper(XGBRegressor(objective ='reg:squarederror', n_estimators = 1, seed = 123, verbosity=0))),
        ("AdaBoostRegressor", ForestWrapper(AdaBoostRegressor(loss ='linear', n_estimators = 6)))
    ]
    ensemble = EnsembleModelRegressor(
        models,
        combiner_model=ForestWrapper(RandomForestRegressor(n_estimators=5, max_leaf_nodes=256, random_state=1, verbose=0))
    )
    ensemble.fit(train_x_df, train_y_df)
    ensemble.save("datasets/model_log.pkl")
    print(ensemble.fitness_)

def analyze_ensemble_model():
    #Load dataset
    print("GATHERING PER-PARTITION SAMPLES")
    FEATURES = list(pd.read_csv("features/features.csv", header=None).iloc[:,0])
    PERFORMANCE = list(pd.read_csv("features/performance.csv", header=None).iloc[:,0])
    train_df = Dataset.load("datasets/train_whole.pkl")
    test_df = Dataset.load("datasets/test_whole.pkl")
    train_x_df,train_y_df = train_df.split(FEATURES,PERFORMANCE)
    test_x_df,test_y_df = test_df.split(FEATURES,PERFORMANCE)

    #Create an ensemble model per-partition
    print("ENSEMBLING")
    ensemble = EnsembleModelRegressor.load("datasets/model_log.pkl")
    print(ensemble.fitnesses_)
    print(ensemble.rmses_)

##############MAIN##################
#Load the performance features and variables
case = 2
model_id = 0

if case == 1:
    df_wrap = Dataset.read_csv("datasets/preprocessed_dataset.csv")
    df_wrap.logp1(features="TOTAL_IO_TIME", base=10)
    partitions = df_wrap.kmeans(features="TOTAL_IO_TIME", k=10, cluster_col="partition").agglomerate(dist_thresh=.5)
    df_wrap.expm1(features="TOTAL_IO_TIME", base=10)
    pp.pprint(partitions.analyze()["TOTAL_IO_TIME"])
    #pd.DataFrame(analysis["TOTAL_IO_TIME"]).transpose().round(decimals=3).to_csv("datasets/partition-stats.csv")

if case == 2:
    stacked_model_per_partition(case = 1)

if case == 3:
    analyze_stacked_model()

if case == 4:
    ensemble_log_regression(case = 1)

if case == 5:
    analyze_ensemble_model()
