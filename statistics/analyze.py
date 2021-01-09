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

from clever.dataset import Dataset, Clusters
from clever.curve_wrapper import CurveWrapper
from clever.forest_wrapper import ForestWrapper
from clever.ensemble_model import EnsembleModelRegressor
from clever.partitioned_model_regressor import PartitionedModelRegressor

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from mord import OrdinalRidge

from sklearn.metrics import f1_score, mean_squared_error as MSE
from scipy import stats
from functools import reduce
from itertools import compress
import pprint, warnings
import pickle

from sklearn.ensemble import StackingRegressor

warnings.simplefilter("ignore") #IGNORE invalid pandas warnings
pp = pprint.PrettyPrinter(depth=6)
SEED = 123

##############HELPER FUNCTIONS##############

def partition_dataset(DATASET, case):
    if case <= 1:
        df_wrap = Dataset().read_csv(DATASET)
        df_wrap.partition("TOTAL_IO_TIME", step=.1, scale=10, min_range=100, cluster_col="partition")
        df_wrap.to_csv("datasets/partitioned_dataset.csv")
    else:
        df_wrap = Dataset().read_csv("datasets/partitioned_dataset.csv")
    partitions = df_wrap.divide(cluster_col="partition")
    return (df_wrap,partitions)

def sample_dataset(partitions,case):
    if case <= 2:
        train_df,hyper_df,test_df = partitions.random_sample(.5)
        train_df.save("datasets/train.pkl")
        #hyper_df.to_csv("datasets/hyper_{}.pkl")
        test_df.save("datasets/test.pkl")
    else:
        train_df = Clusters.load("datasets/train_{}.pkl")
        #hyper_df = Clusters.read_csv("datasets/hyper.pkl")
        test_df = Clusters.load("datasets/test_{}.pkl")
    return train_df,test_df

def test_regression(models, train_df, test_df, features, vars):
    #Get training and testing sets
    train_x = train_df[features]
    train_y = train_df[vars]
    test_x = test_df[features]
    test_y = test_df[vars]

    #Feature Selection
    reduced_model = models.fit(train_x, train_y)

    #Get per-model fitness to the data
    model_fitnesses = {}
    for id,model in models.model.named_estimators_.items():
        score = model.score(test_x,test_y)
        pred = model.predict(test_x)
        rmse = model.rmse(pred, test_y)
        model_fitnesses[id] = {
            "score": score,
            "rmse": rmse
        }

    return (model_fitnesses, reduced_model.feature_importances_)

def stacked_model_per_partition(case):
    #STEP 1: Partition the dataset using TOTAL_IO_TIME
    print("PARTITIONING DATASET")
    df,partitions = partition_dataset("datasets/preprocessed_dataset.csv", case)

    #Load features and performance
    part_model_fitnesses = {}
    FEATURES = list(pd.read_csv("features/features.csv", header=None).iloc[:,0])
    PERFORMANCE = list(pd.read_csv("features/performance.csv", header=None).iloc[:,0])

    #Sample from each partition
    print("GATHERING PER-PARTITION SAMPLES")
    train_dfs,test_dfs = sample_dataset(partitions,case)
    train_x_dfs,train_y_dfs = train_dfs.split(FEATURES,PERFORMANCE)
    test_x_dfs,test_y_dfs = test_dfs.split(FEATURES,PERFORMANCE)

    #Create an ensemble model per-partition
    models = [
        ("RandomForestRegressor", ForestWrapper(RandomForestRegressor(n_estimators=3, max_leaf_nodes=256, random_state=1, verbose=0))),
        ("XGBRegressor", ForestWrapper(XGBRegressor(objective ='reg:squarederror', n_estimators = 1, seed = 123, verbosity=0))),
        ("AdaBoostRegressor", ForestWrapper(AdaBoostRegressor(loss ='linear', n_estimators = 6)))#,
        #("LinearRegression", CurveWrapper(LinearRegression(fit_intercept=False))),
    ]
    ensemble = EnsembleModelRegressor(models, combiner_model=ForestWrapper(RandomForestRegressor(n_estimators=3, max_leaf_nodes=16, random_state=1, verbose=0)))
    model = PartitionedModelRegressor(ensemble, ForestWrapper(RandomForestRegressor(n_estimators=3, max_leaf_nodes=256, random_state=1, verbose=0)))
    model.fit(train_x_dfs, train_y_dfs)
    print(model.fitness_)



##############MAIN##################
#Load the performance features and variables
case = 2
model_id = 0

if case == 0:
    df = Dataset(df = pd.DataFrame(data = [
        {"A":1, "B": "abcv"},
        {"A":2, "B": "ab"},
    ]))
    print(df[df["A"] == 1])

if case == 1:
    df_wrap = Dataset().read_csv("datasets/partitioned_dataset.csv")
    df_wrap.log(features="TOTAL_IO_TIME")
    df_wrap.kmeans(features="TOTAL_IO_TIME", k=9, cluster_col="partition")
    df_wrap.exp(features="TOTAL_IO_TIME")
    analysis = df_wrap.divide(cluster_col="partition").analyze(features="TOTAL_IO_TIME")
    pp.pprint(analysis["TOTAL_IO_TIME"])
    #pd.DataFrame(analysis["TOTAL_IO_TIME"]).transpose().round(decimals=3).to_csv("datasets/partition-stats.csv")

if case == 2:
    stacked_model_per_partition(case = 3)
