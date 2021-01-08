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
import progressbar
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler

from src.curve_wrapper import CurveWrapper
from src.forest_wrapper import ForestWrapper
from src.ensemble_model import EnsembleModelRegressor

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
from sklearn.feature_selection import SequentialFeatureSelector

warnings.simplefilter("ignore") #IGNORE invalid pandas warnings
pp = pprint.PrettyPrinter(depth=6)
SEED = 123

##############HELPER FUNCTIONS##############

def save_model_fitness(model_fitness, path):
    pickle.dump(model_fitness, open( path, "wb" ))

def read_model_fitness(path):
    return pickle.load(open( path, "rb" ))

def ensemble_partition_dataset(DATASET, case):
    if case <= 1:
        df = pd.read_csv(DATASET)
        df = df_partition(df, "TOTAL_IO_TIME", step=.1, scale=10, min_range=100, cluster_col="partition")
        df.to_csv("datasets/partitioned_dataset.csv")
    else:
        df = pd.read_csv("datasets/partitioned_dataset.csv")
    partitions = create_clusters(df, cluster_col="partition")
    return (df,partitions)

def ensemble_sample_dataset(df,partition_id,case):
    if case <= 2:
        #weights, score, train_df, test_df = auto_sample_maker(df, FEATURES, PERFORMANCE, max_split=.75, cluster_col="cluster")
        train_df, test_df = random_sample(df, .5*len(df))
        train_df.to_csv("datasets/train_{}.csv".format(partition_id), index=False)
        test_df.to_csv("datasets/test_{}.csv".format(partition_id), index=False)
        #pp.pprint(weights)
        #print(score)
    else:
        train_df = pd.read_csv("datasets/train_{}.csv".format(partition_id))
        test_df = pd.read_csv("datasets/test_{}.csv".format(partition_id))
    return test_df,train_df

def test_regression(model, train_df, test_df, features, vars):
    #Get training and testing sets
    train_x = train_df[features]
    train_y = train_df[vars]
    test_x = test_df[features]
    test_y = test_df[vars]

    #Feature Selection
    #sfs = SequentialFeatureSelector(model, n_features_to_select=10).fit(train_x, train_y)
    #mask = list(compress(features, sfs.support_))
    #reduced_model = sfs.transform(model)
    reduced_model = model.fit(train_x, train_y)

    #Get the model fitness to the data
    score = reduced_model.score(test_x,test_y)
    pred = reduced_model.predict(test_x)
    rmse = np.sqrt(MSE(test_y, pred))
    return (score, rmse, reduced_model.feature_importances_)

def stacked_model_per_partition(case):
    models = [
        ("RandomForestRegressor", ForestWrapper(RandomForestRegressor(n_estimators=3, max_leaf_nodes=256, random_state=1, verbose=0))),
        ("XGBRegressor", ForestWrapper(XGBRegressor(objective ='reg:squarederror', n_estimators = 1, seed = 123, verbosity=0))),
        ("AdaBoostRegressor", ForestWrapper(AdaBoostRegressor(loss ='linear', n_estimators = 6))),
        ("LinearRegression", CurveWrapper(LinearRegression(fit_intercept=False))),
    ]
    model = EnsembleModelRegressor(models, combiner_model=ForestWrapper(RandomForestRegressor(n_estimators=3, max_leaf_nodes=16, random_state=1, verbose=0)))
    #model = ForestWrapper(RandomForestRegressor(n_estimators=3, max_leaf_nodes=256, random_state=1, verbose=0))

    #STEP 1: Partition the dataset using TOTAL_IO_TIME
    print("PARTITIONING DATASET")
    df,partitions = ensemble_partition_dataset("datasets/preprocessed_dataset.csv", case)

    #Load features and performance
    part_model_fitnesses = {}
    FEATURES = list(pd.read_csv("features/features.csv", header=None).iloc[:,0])
    PERFORMANCE = list(pd.read_csv("features/performance.csv", header=None).iloc[:,0])

    for partition_id, partition_df in partitions.items():
        #STEP 2: Identify optimal weights for stratified random sample
        print("GATHERING SAMPLE FOR PARTITION {}".format(partition_id))
        test_df,train_df = ensemble_sample_dataset(partition_df,partition_id,case)

        #STEP 3: Run models over the sample
        print("Ensembling")
        model_fitness = test_regression(model, train_df, test_df, FEATURES, PERFORMANCE)
        save_model_fitness(model_fitness, "datasets/model/ens_importances_{}.pkl".format(partition_id))
        pp.pprint(model_fitness)




##############MAIN##################
#Load the performance features and variables
case = 3
model_id = 0

if case == -1:
    DATASET="datasets/preprocessed_dataset.csv"
    df = pd.read_csv(DATASET)
    df = df_partition(df, "TOTAL_IO_TIME", step=.1, scale=10, min_range=100, cluster_col="cluster")
    analysis = analyze_clusters(create_clusters(df), ["TOTAL_IO_TIME"])
    pp.pprint(pd.DataFrame(analysis["TOTAL_IO_TIME"]))

if case == -2:
    DATASET="datasets/partitioned_dataset.csv"
    df = pd.read_csv(DATASET)
    analysis = analyze_clusters(create_clusters(df, cluster_col="partition"), ["TOTAL_IO_TIME"])
    pp.pprint(pd.DataFrame(analysis["TOTAL_IO_TIME"]))
    pd.DataFrame(analysis["TOTAL_IO_TIME"]).transpose().round(decimal=3).to_csv("datasets/partition-stats.csv")

if case == 1:
    ensemble_feature_importances_parted(case = 3)

if case == 2:
    view_feature_importances_parted()

if case == 3:
    stacked_model_per_partition(case = 3)
