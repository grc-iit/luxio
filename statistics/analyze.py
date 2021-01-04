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
from sklearn.feature_selection import RFE
from sklearn.metrics import f1_score, mean_squared_error as MSE
from sklearn.tree import export_graphviz
from functools import reduce
import pprint, warnings
import pydot

warnings.simplefilter("ignore") #IGNORE invalid pandas warnings
pp = pprint.PrettyPrinter(depth=6)

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

def visualize_inertias(inertias:dict) -> int:

    """
    A simple line plot that shows the inertia (Sum of Squares) for varying values of k
    (from KMeans).

    INPUT:
        inertias: A dictionary where keys are "k" (number of clusters) and values are "inertias"
    """

    plt.plot(list(inertias.keys()), list(inertias.values()), 'bx-')
    plt.xlabel('k')
    plt.ylabel('Inertia')
    plt.title('Elbow Method For Optimal k')
    plt.show()
    plt.close()
    k = int(input("What is the optimal k?: "))
    return k

def visualize_cdf(df:pd.DataFrame, feature, quantile=1, n_bins=1000, out=None):
    df = df[df[feature] <= df[feature].quantile(quantile)]
    n, bins, patches = plt.hist(df[feature], n_bins, density=True, histtype='step', cumulative=True, label='Empirical')
    plt.xlabel(feature)
    plt.ylabel("% of data")
    plt.title('CDF of {}'.format(feature))
    if out == None:
        plt.show()
    else:
        plt.savefig(out)
    plt.close()

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

        #Visualize inertias
        k = visualize_inertias(inertias)
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

def stratified_random_sample(clusters:dict, n=20) -> pd.DataFrame:

    """
    Selects data from each cluster so that every cluster is equally represented in the dataset.
    If a cluster has less than n data points, it will be extended by repeating entries.
    """

    n = int(n)
    dfs = list(zip(*[random_sample(df,n) for df in clusters.values()]))
    return (pd.concat(dfs[0]), pd.concat(dfs[1]))

def print_importances(importances,features):
    indices = np.argsort(-1*importances)
    sum = 0
    max = sum(importances)
    for feature,importance in [(features[i], importances[i]) for i in indices]:
        sum += importance/max
        print("{}: importance={}, net_variance_explained={}".format(feature,importance/max,sum))

def df_random_forest_classifier(train_df, test_df, features, vars, max_leaf_nodes=None, k=None, visualize=False, score=None) -> tuple:
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

def df_random_forest_regression(train_df, test_df, features, vars, max_leaf_nodes=None, n_trees=10, visualize=False) -> tuple:
    #Get training and testing sets
    train_x = train_df[features]
    train_y = train_df[vars]
    test_x = train_df[features]
    test_y = train_df[vars]

    #Identify clusters of performance and take stratified random sample
    model = RandomForestRegressor(n_estimators=n_trees, random_state=1, verbose=4, max_leaf_nodes=max_leaf_nodes)
    model.fit(train_x,train_y)
    score = model.score(test_df[features],test_df[vars]))
    pred = model.predict(test_x)
    rmse = np.sqrt(MSE(test_y, pred))
    importances = {feature:importance for feature,importance in zip(features,model.feature_importances_)}

    #Visualize the importance of the features
    if visualize:
        print_importances(rf.feature_importances_,features)
        visualize_random_forest(rf,features, cluster_col,"img/rf-reg")

    return (score, importances, rmse)

def df_linreg(train_df, test_df, features, vars) -> tuple:
    #Linear regression
    train_x = RobustScaler().fit_transform(train_df[features])
    test_x = RobustScaler().fit_transform(test_df[features])
    model = LinearRegression(fit_intercept=False).fit(train_x, train_df[vars])
    score = model.score(test_x, test_df[vars])
    pred = model.predict(test_x)
    rmse = np.sqrt(MSE(test_df[vars], pred))
    abscoeff = np.absolute(model.coef_[0])
    importances = {feature:importance for feature,importance in zip(features,abscoeff/sum(abscoeff))}
    return (score, rmse, importances)

def df_ordinal_logistic_regression(train_df, test_df, features, vars):
    #Ordinal Logistic Regression
    train_df[features] = RobustScaler().fit_transform(train_df[features])
    model = OrdinalRidge()
    model.fit(train_df[features], train_df[vars])
    score = model.score(test_df[features], test_df[vars])
    pred = model.predict(test_df[features])
    rmse = np.sqrt(MSE(test_df[vars], pred))
    abscoeff = np.absolute(model.coef_[0])
    importances = {feature:importance for feature,importance in zip(features,abscoeff/sum(abscoeff))}
    return (score, rmse, importances)

def df_xgboost_regression(df:pd.DataFrame, features, vars, n_trees = 10):
    #Gradient Boost Forest Regression
    model = XGBRegressor(objective ='reg:linear', n_estimators = n_trees, seed = 123)
    model.fit(train[features], train[vars])
    score = model.score(test_df[features],test_df[vars]))
    pred = model.predict(test_df[features])
    rmse = np.sqrt(MSE(test_df[vars], pred))
    importances = {feature:importance for feature,importance in zip(features,rf.feature_importances_)}
    return (score, rmse, importances)

def auto_sample_maker():
    """
    Using nonlinear least squares regression, select a stratified sample such that the
    value (1-r^2) is minimized when using RandomForestRegression.
    """

    return




##############MAIN##################

#Load the performance features and variables
FEATURES = list(pd.read_csv("features/features.csv", header=None).iloc[:,0])
PERFORMANCE = list(pd.read_csv("features/performance.csv", header=None).iloc[:,0])

#Load dataset and select features
#DATASET="datasets/preprocessed_dataset.csv"
DATASET="datasets/agglomerative.csv"
df = pd.read_csv(DATASET)
case = 6

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
        visualize_cdf(df, var, quantile=.5, out="img/cdf_{}.png".format(var))
