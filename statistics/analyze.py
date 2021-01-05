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
from scipy.optimize import least_squares, differential_evolution, Bounds
from sklearn.metrics import f1_score, mean_squared_error as MSE
from scipy import stats
from functools import reduce
import pprint, warnings
import pickle
import jenkspy

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
        print(q)
        if q > 1:
            q = 1
        id += 1
    return df

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
    test_x = test_df[features]
    test_y = test_df[vars]

    #Train model
    model = RandomForestClassifier(random_state=1, verbose=4, max_leaf_nodes=max_leaf_nodes)
    model.fit(train_x,train_y)

    #Get the model fitness to the data
    pred = model.predict(test_x)
    score = f1_score(pred, test_y, average=score) #None, 'micro', 'macro', 'weighted'
    importances = {feature:importance for feature,importance in zip(features,model.feature_importances_)}
    return (score, importances, -1)

def df_random_forest_regression(train_df, test_df, features, vars, max_leaf_nodes=None, n_trees=10) -> tuple:
    #Get training and testing sets
    train_x = train_df[features]
    train_y = train_df[vars]
    test_x = test_df[features]
    test_y = test_df[vars]

    #Train model
    model = RandomForestRegressor(n_estimators=n_trees, random_state=1, verbose=4, max_leaf_nodes=max_leaf_nodes)
    model.fit(train_x,train_y)

    #Get the model fitness to the data
    score = model.score(test_x,test_y)
    pred = model.predict(test_x)
    rmse = np.sqrt(MSE(test_y, pred))
    importances = {feature:importance for feature,importance in zip(features,model.feature_importances_)}
    return (score, importances, rmse)

def df_xgboost_regression(train_df, test_df, features, vars, n_trees = 10):
    #Get training and testing sets
    train_x = train_df[features]
    train_y = train_df[vars]
    test_x = test_df[features]
    test_y = test_df[vars]

    #Gradient Boost Forest Regression
    model = XGBRegressor(objective ='reg:linear', n_estimators = n_trees, seed = 123, verbosity=2)
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
    test_x = test_df[features]
    test_y = test_df[vars]

    #Gradient Boost Forest Regression
    model = AdaBoostRegressor(loss ='linear', n_estimators = n_trees)
    model.fit(train_x, train_y)

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
    test_y = test_df[vars]

    #Train model
    model = LinearRegression(fit_intercept=False).fit(train_x, train_y)

    #Get the model fitness to the data
    score = model.score(test_x, test_y)
    pred = model.predict(test_x)
    rmse = np.sqrt(MSE(test_y, pred))
    abscoeff = np.absolute(model.coef_[0])
    abscoeff /= sum(abscoeff)
    importances = {feature:importance for feature,importance in zip(features,abscoeff)}
    return (score, importances, rmse)

def df_ordinal_logistic_regression(train_df, test_df, features, vars):
    #Get training and testing sets
    train_x = RobustScaler().fit_transform(train_df[features])
    train_y = train_df[vars]
    test_x = RobustScaler().fit_transform(test_df[features])
    test_y = test_df[vars]

    #Ordinal Logistic Regression
    model = OrdinalRidge()
    model.fit(train_x, train_y)

    #Get the model fitness to the data
    score = model.score(test_x, test_y)
    pred = model.predict(test_x)
    rmse = np.sqrt(MSE(test_y, pred))
    abscoeff = np.absolute(model.coef_[0])
    abscoeff /= sum(abscoeff)
    importances = {feature:importance for feature,importance in zip(features,abscoeff)}
    return (score, importances, rmse)

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

def sample_score_fun(x:list, clusters:dict, features:list, vars:list):
    train_df,test_df = stratified_random_sample(clusters, x)
    score,importances,rmse = df_random_forest_regression(train_df, test_df, features, vars, max_leaf_nodes=64, n_trees=1)
    return 1 - score

def auto_sample_maker(df:pd.DataFrame, features:list, vars:list, max_split=.75, cluster_col="cluster") -> tuple:
    """
    Using nonlinear least squares regression, select a stratified sample such that the
    value (1-r^2) is minimized when using RandomForestRegression.
    """

    clusters = create_clusters(df, cluster_col=cluster_col)
    k = len(df[cluster_col].unique())
    x0 = [.5 for i in range(k)]
    res = least_squares(sample_score_fun, x0, args=(clusters, features, vars), bounds=(0,max_split), max_nfev=10, diff_step=[10]*k)
    #res = differential_evolution(sample_score_fun, [(0,max_split)]*k, args=(clusters, features, vars), maxiter=3, popsize=1, workers=4)
    weights = res.x
    train_df,test_df = stratified_random_sample(clusters, weights)
    score,importances,rmse = df_random_forest_regression(train_df, test_df, features, vars, max_leaf_nodes=256, n_trees=3)
    return (weights, score, train_df, test_df)

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

def ensemble_feature_importances(train_df, test_df, features, vars, case = 1):

    #LOAD CHECKPOINT DATA
    if case == 1:
        DATASET="datasets/preprocessed_dataset.csv"
        df = pd.read_csv(DATASET)
    if case == 2:
        DATASET="datasets/partitioned_dataset.csv"
        df = pd.read_csv(DATASET)
    if (case == 2) or (case == 3):
        FEATURES = list(pd.read_csv("features/features.csv", header=None).iloc[:,0])
        PERFORMANCE = list(pd.read_csv("features/performance.csv", header=None).iloc[:,0])
        partitions = create_clusters(df, cluster_col="partition")
    if case == 3:
        train_df = pd.read_csv("datasets/train.csv")
        test_df = pd.read_csv("datasets/test.csv")
    if case == 4:
        model_fitnesses = [read_model_fitness("datasets/model/importances_{}.pkl".format(model_id)) for model_id in range(5)]

    #STEP 1: Partition the dataset using TOTAL_IO_TIME
    if case <= 1:
        df = df_partition(df, "TOTAL_IO_TIME", step=.1, scale=10, min_range=100, cluster_col="partition")
        df.to_csv("datasets/partitioned_dataset.csv")

    for partition_id, partition_df in partitions.items():
        #STEP 2: Identify optimal weights for stratified random sample
        if case <= 2:
            #weights, score, train_df, test_df = auto_sample_maker(df, FEATURES, PERFORMANCE, max_split=.75, cluster_col="cluster")
            train_df, test_df = random_sample(df, .5*len(df))
            train_df.to_csv("datasets/train.csv", index=False)
            test_df.to_csv("datasets/test.csv", index=False)
            #pp.pprint(weights)
            #print(score)

        #STEP 3: Run models over the sample
        if case <= 3:
            model_fitnesses = []
            for model_id in range(5):
                model_fitness = ensemble_feature_importances(train_df, test_df, FEATURES, PERFORMANCE, model_id)
                save_model_fitness(model_fitness, "datasets/model/importances_{}.pkl".format(model_id))
                model_fitnesses.append(model_fitness)

    #STEP 4: Combine the feature importances
    if case <= 4:
        pp.pprint({model_id:model_fitness[0] for model_id,model_fitness in enumerate(model_fitnesses)})
        pp.pprint(combine_feature_importances(model_fitnesses))

def combine_feature_importances(model_fitnesses:list, thresh=.9):
    avg_importances = {}
    net_score = 0
    #Get total weighted importance of feature for each model
    for score,importances,rmse in model_fitnesses:
        for key,value in importances.items():
            if key not in avg_importances:
                avg_importances[key] = 0
            if score < 0:
                score = 0
            avg_importances[key] += score*value
        net_score += score
    #Get average
    for key,value in avg_importances.items():
        avg_importances[key] /= net_score
    #Sort by importance
    importances = list(avg_importances.items())
    importances.sort(key = lambda x: x[1], reverse=True)
    return importances


##############MAIN##################
#Load the performance features and variables
case = -2
model_id = 0

#Cumulative Distribution Function on each PERFORMANCE variable
if case == -1:
    DATASET="datasets/preprocessed_dataset.csv"
    df = pd.read_csv(DATASET)
    df = df[df["TOTAL_IO_TIME"] <= df["TOTAL_IO_TIME"].quantile(.99)]
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

if case == -2:
    DATASET="datasets/preprocessed_dataset.csv"
    df = pd.read_csv(DATASET)
    df = df_partition(df, "TOTAL_IO_TIME", step=.1, scale=10, cluster_col="cluster")
    pp.pprint(analyze_clusters(create_clusters(df), ["TOTAL_IO_TIME"]))

#STEP 1: Partition the dataset using TOTAL_IO_TIME
elif case == 1:
    DATASET="datasets/preprocessed_dataset.csv"
    df = pd.read_csv(DATASET)
    df = df_partition(df, "TOTAL_IO_TIME", step=.1, scale=10, min_range=100, cluster_col="cluster")
    df.to_csv("datasets/partitioned_dataset.csv")

#STEP 2: Identify optimal weights for stratified random sample
elif case == 2:
    FEATURES = list(pd.read_csv("features/features.csv", header=None).iloc[:,0])
    PERFORMANCE = list(pd.read_csv("features/performance.csv", header=None).iloc[:,0])
    DATASET="datasets/partitioned_dataset.csv"
    df = pd.read_csv(DATASET)
    weights, score, train_df, test_df = auto_sample_maker(df, FEATURES, PERFORMANCE, max_split=.75, cluster_col="cluster")
    train_df.to_csv("datasets/train.csv", index=False)
    test_df.to_csv("datasets/test.csv", index=False)
    pp.pprint(weights)
    print(score)

#STEP 3: Run models over the sample
elif case == 3:
    FEATURES = list(pd.read_csv("features/features.csv", header=None).iloc[:,0])
    PERFORMANCE = list(pd.read_csv("features/performance.csv", header=None).iloc[:,0])
    train_df = pd.read_csv("datasets/train.csv")
    test_df = pd.read_csv("datasets/test.csv")
    model_fitness = ensemble_feature_importances(train_df, test_df, FEATURES, PERFORMANCE, model_id)
    save_model_fitness(model_fitness, "datasets/model/importances_{}.pkl".format(model_id))

#STEP 4: Combine the feature importances
elif case == 4:
    model_fitnesses = [read_model_fitness("datasets/model/importances_{}.pkl".format(model_id)) for model_id in range(5)]
    pp.pprint({model_id:model_fitness[0] for model_id,model_fitness in enumerate(model_fitnesses)})
    pp.pprint(combine_feature_importances(model_fitnesses))
