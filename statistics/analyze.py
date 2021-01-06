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

from src import curve_wrapper, forest_wrapper, ensemble_model
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from mord import OrdinalRidge

from sklearn.metrics import f1_score, mean_squared_error as MSE
from scipy import stats
from functools import reduce
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

def basic_stats(df:pd.DataFrame, n=None) -> dict:

    """
    Calculates basic statistics for a dataframe

    INPUT:
        df: A pandas dataframe containing "FEATURES" and "PERFORMANCE"
    OUTPUT:
        A dictionary containing basic statistics for the df: Mean, Std Dev, IQR
    """

    if n == None:
        n = len(df)

    return {
        "n": len(df),
        "rel_n": len(df)/n,
        "mean": df.mean(),
        "std": df.std(),
        "0%": df.min(),
        "25%": df.quantile(q=.25),
        "50%": df.median(),
        "75%": df.quantile(q=.75),
        "90%": df.quantile(q=.9),
        "100%": df.max()
    }

def create_clusters(df:pd.DataFrame, cluster_col="cluster") -> dict:

    """
    Divides the dataframe into clusters based on the "cluster" key.

    INPUT:
        df: A pandas dataframe containing the "cluster" key
    OUTPUT:
        clusters: A dictionary where the key is cluster_id and the value is
        a pandas dataframe of entries representing that cluster
    """

    df.sort_values(by=cluster_col,inplace=True)
    cluster_ids = df[cluster_col].unique().tolist()
    clusters = { cluster_id:df[df[cluster_col] == cluster_id] for cluster_id in cluster_ids }
    return clusters

def analyze_clusters(clusters:dict, features) -> dict:

    """
    Calculates the basic statistics for each cluster for "PERFORMANCE" variables:
    Mean, Std Dev, IQR

    INPUT:
        clusters: A dictionary where keys are cluster IDs and values are dfs
    OUTPUT:
        stats: A dictionary where keys are cluster IDs and values are basic stats.
    """

    net_len = sum([len(df) for df in clusters.values()])
    return { var: {cluster_id:basic_stats(df[var], net_len) for cluster_id,df in clusters.items()} for var in features }

def cluster_sizes(clusters:dict, features) -> dict:
    net_len = sum([len(df) for df in clusters.values()])
    return { cluster_id : {"n" : len(df), "rel_n" : len(df)/net_len } for cluster_id,df in clusters.items() }

def print_clusters(clusters:dict) -> None:

    """
    Prints clusters

    INPUT:
        clusters: A dictionary where keys are cluster IDs and values are dfs
    """

    for cluster_id,df in clusters.items():
        print("CLUSTER: {}".format(cluster_id))
        print(df)
        print()

def df_partition(df:pd.DataFrame, feature, step=.1, scale=10, min_range=100, cluster_col="cluster"):

    """
    Partition dataset by quantile.
    """

    feature_df = df[feature]
    q = step
    min_value = feature_df.min()
    df[cluster_col] = 0
    id = 0
    while True:
        max_value = feature_df.quantile(q)*scale
        if (max_value - min_value) < min_range:
            max_value = min_value + min_range
        df.loc[(min_value <= df[feature]) & (df[feature] < max_value),cluster_col] = id
        if q == 1:
            break
        min_value = max_value
        q = stats.percentileofscore(feature_df,max_value)/100+step
        if q > 1:
            q = 1
        id += 1
    return df

def df_kmeans(df:pd.DataFrame, features, k=None, cluster_col="cluster", return_centers=False) -> pd.DataFrame:
    """
    Group rows of pandas dataframe using KMeans based on features

    INPUT:
        df: A pandas dataframe containing "features"
        features: The set of features to use KMeans on
        k: The number of clusters to create
        cluster_col: The name of the column to add to the dataframe
    OUTPUT:
        df: The same pandas dataframe except with "cluster_col" column
    """

    #Standardize Dataframe Features
    feature_df = RobustScaler().fit_transform(df[features])

    #Create different number of clusters
    clusters = {}
    inertias = {}
    centers = {}
    if k == None:
        for k in progressbar.progressbar([2, 4, 6, 8, 12]):
            if len(feature_df) < k:
                inertias[k] = np.inf
                clusters[k] = None
                continue
            km = KMeans(n_clusters=k, verbose=10)
            clusters[k] = np.array(km.fit_predict(feature_df))
            inertias[k] = km.inertia_
            centers[k] = km.cluster_centers_
    else:
        if len(feature_df) < k:
            inertias[k] = np.inf
            clusters[k] = None
            centers[k] = None
        else:
            km = KMeans(n_clusters=k, verbose=10)
            clusters[k] = np.array(km.fit_predict(feature_df))
            inertias[k] = km.inertia_
            centers[k] = km.cluster_centers_

    #Cluster using optimal clustering
    df[cluster_col] = clusters[k]
    if not return_centers:
        return df
    else:
        return (df, centers[k])

def df_agglomerative(df:pd.DataFrame, features, max_k = 200, dist_thresh=None, cluster_col="cluster") -> pd.DataFrame:

    """
    Runs agglomerative clustering on the clusters that result from KMeans to group features.

    INPUT:
        df: A pandas dataframe containing "features"
        features: The set of features to use KMeans on
        max_k: The maximum number of clusters to create
        cluster_col: The name of the column to add to the dataframe
    OUTPUT:
        df: The same pandas dataframe except with "cluster_col" column
    """

    #A simple distance threshold estimate
    if dist_thresh==None:
        dist_thresh = np.sqrt(len(features)/4)

    #Run KMeans with high k and extract cluster centers
    df,centers = df_kmeans(df, features, max_k, cluster_col="$tempcol", return_centers=True)
    centers = pd.DataFrame(data=centers)

    #Run agglomerative clustering on the cluster centers
    agg = AgglomerativeClustering(n_clusters=None, distance_threshold=dist_thresh).fit(centers)
    labels = agg.labels_
    k = agg.n_clusters_

    #Re-label each cluster
    df[cluster_col] = 0
    for cluster, label in zip(range(max_k), labels):
        df.loc[df["$tempcol"] == cluster, cluster_col] = label
    df = df.drop(columns="$tempcol")
    return df

def print_importances(importances,features):
    indices = np.argsort(-1*importances)
    sum = 0
    max = sum(importances)
    for feature,importance in [(features[i], importances[i]) for i in indices]:
        sum += importance/max
        print("{}: importance={}, net_variance_explained={}".format(feature,importance/max,sum))

def random_sample(df:pd.DataFrame, n) -> pd.DataFrame:

    """
    Randomly sample data from a dataframe.
    It samples without replacement if n is smaller than the dataset.
    """

    n = int(n)
    if len(df) > n:
        sample = df.sample(n, replace=False)
        return (sample, df.drop(sample.index))
    else:
        return (df.sample(n, replace=True), df)

def stratified_random_sample(clusters:dict, weights:list) -> tuple:

    """
    Weighted stratified random sample
    """

    dfs = list(zip(*[random_sample(df,len(df)*weight) for df,weight in zip(clusters.values(),weights)]))
    return (pd.concat(dfs[0]), pd.concat(dfs[1]))

def ensemble_model(train_df, test_df, features, vars, model_id):
    if model_id == 0:
        return df_random_forest_regression(train_df, test_df, features, vars, max_leaf_nodes=256, n_trees=3)
    elif model_id == 1:
        return df_xgboost_regression(train_df, test_df, features, vars)
    elif model_id == 2:
        return df_adaboost_regression(train_df, test_df, features, vars)
    elif model_id == 3:
        return df_linreg(train_df, test_df, features, vars)
    elif model_id == 4:
        return df_ordinal_logistic_regression(train_df, test_df, features, vars)

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

def ensemble_feature_importances(case = 1):
    #STEP 1: Partition the dataset using TOTAL_IO_TIME
    print("PARTITIONING DATASET")
    df,partitions = ensemble_partition_dataset("datasets/preprocessed_dataset.csv", case)

    #Load features and performance
    part_model_fitnesses = {}
    FEATURES = list(pd.read_csv("features/features.csv", header=None).iloc[:,0])
    PERFORMANCE = list(pd.read_csv("features/performance.csv", header=None).iloc[:,0])

    #STEP 2: Identify optimal weights for stratified random sample
    print("GATHERING SAMPLE FOR PARTITION {}".format(0))
    test_df,train_df = ensemble_sample_dataset(df,0,case)

    #STEP 3: Run models over the sample
    for model_id in range(5):
        print("MODEL {}".format(model_id))
        model_fitness = ensemble_model(train_df, test_df, FEATURES, PERFORMANCE, model_id)
        for partition_id, partition_df in partitions.items():
            model_fitnesses = {} #Think harder
            save_model_fitness(model_fitness, "datasets/model/importances_{}_{}.pkl".format(partition_id, model_id))

def ensemble_feature_importances_parted(case = 1):
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
        model_fitnesses = {}
        for model_id in range(5):
            print("ENSEMBLE {}".format(model_id))
            model_fitness = ensemble_model(train_df, test_df, FEATURES, PERFORMANCE, model_id)
            save_model_fitness(model_fitness, "datasets/model/importances_{}_{}.pkl".format(partition_id, model_id))

def weighted_avg(model_fitnesses):
    model_scores = pd.DataFrame([score if score >= 0 else 0 for score,importances,rmse in model_fitnesses.values()]).transpose() #1 x M
    model_importances = pd.DataFrame([importances for score,importances,rmse in model_fitnesses.values()]) #M x F
    weighted_dot = model_scores.dot(model_importances) #1 x F
    net_score = sum(model_scores)
    max_score =  max(model_scores)
    weighted_dot = ((weighted_dot/net_score)*max_score)
    return weighted_dot #1 x F

def max_weighted_avg(model_fitnesses):
    model_scores = pd.DataFrame([[score if score >= 0 else 0 for score,importances,rmse in model_fitnesses.values()] for i in range(len(model_fitnesses.values()))]) #M x M
    model_importances = pd.DataFrame([importances for score,importances,rmse in model_fitnesses.values()]) #M x F
    weighted_dot = model_scores.dot(model_importances) #M x F
    max_df = pd.DataFrame({feature:weighted_dot[feature].max() for feature in weighted_dot.columns}, index=[0])
    return max_df #1 x F

def combine_feature_importances_parted(part_model_fitnesses:dict):
    importances = {}

    #Weighted-Weighted Average
    result = sum([weighted_avg(model_fitnesses) for model_fitnesses in part_model_fitnesses.values()])/len(part_model_fitnesses.values())
    importances["weighted"] = [(col, result[col][0]) for col in result.columns]
    importances["weighted"].sort(key = lambda pair : pair[1], reverse=True)

    #Max-Weighted Average
    result = sum([max_weighted_avg(model_fitnesses) for model_fitnesses in part_model_fitnesses.values()])/len(part_model_fitnesses.values())
    importances["max_avg"] = [(col, result[col][0]) for col in result.columns]
    importances["max_avg"].sort(key = lambda pair : pair[1], reverse=True)

    return importances

def view_feature_importances_parted():
    DATASET="datasets/partitioned_dataset.csv"
    #df = pd.read_csv(DATASET)
    #partitions = create_clusters(df, cluster_col="partition")
    partitions = {i : True for i in range(4)}
    part_model_fitnesses = {partition_id : {model_id : read_model_fitness("datasets/model/importances_{}_{}.pkl".format(partition_id, model_id)) for model_id in range(5)} for partition_id in partitions.keys()}
    for partition_id, model_fitnesses in part_model_fitnesses.items():
        print("---------PARTITION {}-------------".format(partition_id))
        pp.pprint({model_id:(model_fitness[0], model_fitness[2]) for model_id,model_fitness in model_fitnesses.items()})
        print("----------------------------------")
    pp.pprint(combine_feature_importances_parted(part_model_fitnesses))



from itertools import compress

def test_regression(model, train_df, test_df, features, vars):
    #Get training and testing sets
    train_x = train_df[features]
    train_y = train_df[vars]
    test_x = test_df[features]
    test_y = test_df[vars]

    #Feature Selection
    sfs = SequentialFeatureSelector(model, n_features_to_select=10).fit(train_x, train_y)
    mask = list(compress(features, sfs.support_))
    reduced_model = sfs.transform(model)

    #Get the model fitness to the data
    score = reduced_model.score(test_x,test_y)
    pred = reduced_model.predict(test_x)
    rmse = np.sqrt(MSE(test_y, pred))
    return (score, rmse, mask)

def stacked_model_per_partition(case):
    models = [
        ("RandomForestRegressor", RandomForestRegressor(n_estimators=3, max_leaf_nodes=256, random_state=1, verbose=4)),
        ("XGBRegressor", XGBRegressor(objective ='reg:squarederror', n_estimators = 5, seed = 123, verbosity=2)),
        ("AdaBoostRegressor", AdaBoostRegressor(loss ='linear', n_estimators = 6)),
        ("LinearRegression", LinearRegression(fit_intercept=False)),
    ]
    ensemble = EnsembleModelRegressor(models)

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
        ensemble.fit(train_df[FEATURES], train_df[PERFORMANCE])
        return

        #STEP 3: Run models over the sample
        print("Ensembling")
        model_fitness = test_regression(ensemble, train_df, test_df, FEATURES, PERFORMANCE)
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
