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

import pandas as pd
import progressbar
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from mord import OrdinalRidge
from xgboost import XGBRegressor
from scipy.optimize import least_squares
from sklearn.feature_selection import RFE
from sklearn.metrics import f1_score, mean_squared_error as MSE
from sklearn.tree import export_graphviz
from functools import reduce
import pprint, warnings
import pydot

warnings.simplefilter("ignore") #IGNORE invalid pandas warnings
pp = pprint.PrettyPrinter(depth=6)
SEED = 123

##############HELPER FUNCTIONS##############

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

def stratified_random_sample(clusters:dict, weights:list) -> pd.DataFrame:

    """
    Weighted stratified random sample
    """

    dfs = list(zip(*[random_sample(df,len(df)*weight) for df,weight in zip(clusters.values(),weights)]))
    return (pd.concat(dfs[0]), pd.concat(dfs[1]))

def print_importances(importances,features):
    indices = np.argsort(-1*importances)
    sum = 0
    max = sum(importances)
    for feature,importance in [(features[i], importances[i]) for i in indices]:
        sum += importance/max
        print("{}: importance={}, net_variance_explained={}".format(feature,importance/max,sum))

def df_random_forest_classifier(train_df, test_df, features, vars, max_leaf_nodes=None, k=None, score=None) -> tuple:
    #Get training and testing sets
    train_x = train_df[features]
    train_y = train_df[vars]
    test_x = train_df[features]
    test_y = train_df[vars]

    #Train model
    model = RandomForestClassifier(random_state=1, verbose=4, max_leaf_nodes=max_leaf_nodes)
    model.fit(train_x,train_y)

    #Get the model fitness to the data
    pred = model.predict(test_x)
    score = f1_score(pred, test_y, average=score) #None, 'micro', 'macro', 'weighted'
    importances = {feature:importance for feature,importance in zip(features,model.feature_importances_)}
    return (score, importances)

def df_random_forest_regression(train_df, test_df, features, vars, max_leaf_nodes=None, n_trees=10) -> tuple:
    #Get training and testing sets
    train_x = train_df[features]
    train_y = train_df[vars]
    test_x = train_df[features]
    test_y = train_df[vars]

    #Train model
    model = RandomForestRegressor(n_estimators=n_trees, random_state=1, verbose=4, max_leaf_nodes=max_leaf_nodes)
    model.fit(train_x,train_y)

    #Get the model fitness to the data
    score = model.score(test_x,test_y)
    pred = model.predict(test_x)
    rmse = np.sqrt(MSE(test_y, pred))
    importances = {feature:importance for feature,importance in zip(features,model.feature_importances_)}
    return (score, importances, rmse)

def df_linreg(train_df, test_df, features, vars) -> tuple:
    #Get training and testing sets
    train_x = RobustScaler().fit_transform(train_df[features])
    train_y = train_df[vars]
    test_x = RobustScaler().fit_transform(test_df[features])
    test_y = train_df[vars]

    #Train model
    model = LinearRegression(fit_intercept=False).fit(train_x, train_y)

    #Get the model fitness to the data
    score = model.score(test_x, test_y)
    pred = model.predict(test_x)
    rmse = np.sqrt(MSE(test_y, pred))
    abscoeff = np.absolute(model.coef_[0])
    importances = {feature:importance for feature,importance in zip(features,abscoeff/sum(abscoeff))}
    return (score, importances, rmse)

def df_ordinal_logistic_regression(train_df, test_df, features, vars):
    #Get training and testing sets
    train_x = RobustScaler().fit_transform(train_df[features])
    train_y = train_df[vars]
    test_x = RobustScaler().fit_transform(test_df[features])
    test_y = train_df[vars]

    #Ordinal Logistic Regression
    model = OrdinalRidge()
    model.fit(train_x, train_y)

    #Get the model fitness to the data
    score = model.score(test_x, test_y)
    pred = model.predict(test_x)
    rmse = np.sqrt(MSE(test_y, pred))
    abscoeff = np.absolute(model.coef_[0])
    importances = {feature:importance for feature,importance in zip(features,abscoeff/sum(abscoeff))}
    return (score, importances, rmse)

def df_xgboost_regression(train_df, test_df, features, vars, n_trees = 10):
    #Get training and testing sets
    train_x = train_df[features]
    train_y = train_df[vars]
    test_x = train_df[features]
    test_y = train_df[vars]

    #Gradient Boost Forest Regression
    model = XGBRegressor(objective ='reg:linear', n_estimators = n_trees, seed = 123)
    model.fit(train_x, train_y)

    #Get the model fitness to the data
    score = model.score(test_x,test_y)
    pred = model.predict(test_x)
    rmse = np.sqrt(MSE(test_y, pred))
    importances = {feature:importance for feature,importance in zip(features,model.feature_importances_)}
    return (score, importances, rmse)

def df_adaboost_regression(train_df, test_df, features, vars, n_trees = 10):
    #Get training and testing sets
    train_x = train_df[features]
    train_y = train_df[vars]
    test_x = train_df[features]
    test_y = train_df[vars]

    #Gradient Boost Forest Regression
    model = AdaBoostRegressor(loss ='linear', n_estimators = n_trees, seed = 123)
    model.fit(train_x, train_y)

    #Get the model fitness to the data
    score = model.score(test_x,test_y)
    pred = model.predict(test_x)
    rmse = np.sqrt(MSE(test_y, pred))
    importances = {feature:importance for feature,importance in zip(features,model.feature_importances_)}
    return (score, importances, rmse)

def sample_score_fun(x:list, clusters:dict):
    train_df,test_df = stratified_random_sample(clusters, x)
    score,importances,rmse = df_random_forest_regression(train_df, test_df, features, vars, max_leaf_nodes=256, n_trees=3)
    return 1 - score

def auto_sample_maker(df:pd.DataFrame, max_split=.75, cluster_col="cluster"):
    """
    Using nonlinear least squares regression, select a stratified sample such that the
    value (1-r^2) is minimized when using RandomForestRegression.
    """

    clusters = create_clusters(df, cluster_col=cluster_col)
    k = len(df[cluster_col].unique())
    x0 = [max_split/k]*k
    model = least_squares(sample_score_fun, x0, args=(clusters))
    return




##############MAIN##################

#Load the performance features and variables
FEATURES = list(pd.read_csv("features/features.csv", header=None).iloc[:,0])
PERFORMANCE = list(pd.read_csv("features/performance.csv", header=None).iloc[:,0])

#Load dataset and select features
#DATASET="datasets/preprocessed_dataset.csv"
DATASET="datasets/agglomerative.csv"
df = pd.read_csv(DATASET)
case = 7

#STEP 1: Partition The Dataset on IO_TIME
#STEP 2: Learn the Best Proportions of Stratified Random Sampling
#STEP 3: Use the sample to reduce features using an ensemble of different models

#Agglomerative Clustering on PERFORMANCE
if case == 3:
    df = df_agglomerative(df, features=PERFORMANCE, max_k=400, cluster_col="cluster")
    clusters = create_clusters(df)
    pp.pprint(analyze_clusters(clusters, features=PERFORMANCE))
    df.to_csv("datasets/agglomerative.csv", index=False)

#Agglomerative Clustering on each PERFORMANCE variable
elif case == 4:
    for var in PERFORMANCE:
        df = df_agglomerative(df, features=[var], max_k=200, cluster_col="cluster_{}".format(var))
    df.to_csv("datasets/agglomerative.csv", index=False)

#Analyze clusters for each performance variable
elif case == 5:
    for var in PERFORMANCE:
        clusters = create_clusters(df,cluster_col="cluster_{}".format(var))
        pp.pprint(analyze_clusters(clusters, features=[var]))

#Cumulative Distribution Function on each PERFORMANCE variable
elif case == 6:
    for var in PERFORMANCE:
        visualize_cdf(df, var, quantile=.6, out="img/cdf_{}.png".format(var))
elif case == 7:
    quantiles = [(0, .2), (.2,.35), (.35, .5), (.5,.7), (.7,.8), (.8, 1)]
    for var in ["TOTAL_IO_TIME"]:
        print("-------{}----------".format(var))
        for qs in quantiles:

            QR = df[(df[var].quantile(qs[0]) <= df[var]) & (df[var] <= df[var].quantile(qs[1]))]
            print("{} - {} fraction of data".format(qs[0], qs[1]))
            pp.pprint(basic_stats(QR[var], n=len(df)))
        print("--------------------")
        print()
        print()
