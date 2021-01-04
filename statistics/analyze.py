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
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from mord import OrdinalRidge
from sklearn.feature_selection import RFE
from sklearn.metrics import f1_score
from sklearn.tree import export_graphviz
from functools import reduce
import pprint, warnings
import pydot

warnings.simplefilter("ignore") #IGNORE invalid pandas warnings
pp = pprint.PrettyPrinter(depth=6)

##############HELPER FUNCTIONS##############

def df_pca(df:pd.DataFrame, features) -> None:

    """
    Attempt to find the most influential factors explaining performance

    INPUT:
        df: A pandas dataframe containing "FEATURES"
    """

    if len(df) < 3:
        return
    df = StandardScaler().fit_transform(df[features])
    pca_fit= PCA().fit(df)
    print("n: {}".format(len(df)))
    print(pca_fit.explained_variance_ratio_)
    #print(pca_fit.transform(df))

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

def df_gauss_mixtures(df:pd.DataFrame, features,k=None, cluster_col="cluster") -> pd.DataFrame:

    """
    Group rows of pandas dataframe using Gaussian Mixture Models based on features

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
    if k == None:
        for k in progressbar.progressbar([2, 4, 6, 8, 12]):
            if len(feature_df) < k:
                inertias[k] = np.inf
                clusters[k] = None
                continue
            gmm = GaussianMixture(n_components=k, verbose=10)
            clusters[k] = np.array(gmm.fit_predict(feature_df))
            df[cluster_col] = clusters[k]
            stats = analyze_clusters(create_clusters(df), features=features)
            inertias[k] = sum([cluster["std"]**2 for var in stats.values() for cluster in var.values()])

        #Visualize Inertias
        k = visualize_inertias(inertias)
    else:
        if len(feature_df) < k:
            inertias[k] = np.inf
            clusters[k] = None
        else:
            gmm = GaussianMixture(n_components=k, verbose=10)
            clusters[k] = np.array(gmm.fit_predict(feature_df))
            df[cluster_col] = clusters[k]
            stats = analyze_clusters(create_clusters(df), features=features)
            inertias[k] = sum([cluster["std"]**2 for var in stats.values() for cluster in var.values()])

    #Cluster using optimal clustering
    df[cluster_col] = clusters[k]
    return df

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

def pca_clusters(clusters:dict, features) -> dict:

    """
    Performs PCA for each cluster.

    INPUT:
        clusters: A dictionary where keys are cluster IDs and values are dfs
    OUTPUT:
        stats: A dictionary where keys are cluster IDs and values are stats dicts.
    """

    for cluster_id,df in clusters.items():
        print("CLUSTER: {}".format(cluster_id))
        df_pca(df[features], features=features)

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
        #return (df.sample(n, replace=True), pd.DataFrame(columns=df.columns))

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
    for feature,importance in [(features[i], importances[i]) for i in indices]:
        sum += importance
        print("{}: importance={}, net_variance_explained={}".format(feature,importance,sum))

def visualize_random_forest_features(rf,features):
    importances = rf.feature_importances_
    indices = np.argsort(importances)
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()
    plt.close()

def visualize_random_forest(rf,features,vars,out):
    print(len(rf.estimators_))
    export_graphviz(rf.estimators_[-1],
                out_file=out + ".dot",
                feature_names = features,
                class_names = vars,
                rounded = True, proportion = False,
                precision = 2, filled = True)
    (graph,) = pydot.graph_from_dot_file(out + ".dot")
    graph.write_png(out + ".png")

def df_random_forest_classifier(df:pd.DataFrame, features, vars, max_leaf_nodes=None, k=None, cluster_col="cluster"):
    #Identify clusters of performance and take stratified random sample
    clusters = create_clusters(df)
    df = stratified_random_sample(clusters, n=20)

    #Use features to identify classes
    rf = RandomForestClassifier(random_state=1, verbose=4, max_leaf_nodes=max_leaf_nodes)
    rf.fit(df[features],df["cluster"])

    #Get the model fitness to the data
    pred = rf.predict(df[features])
    pp.pprint("Accuracy: {}".format(rf.score(df[features], df["cluster"])))
    print("F1 (None): {}".format(f1_score(pred, df["cluster"], average=None)))
    print("F1 (Micro): {}".format(f1_score(pred, df["cluster"], average='micro')))
    print("F1 (Macro): {}".format(f1_score(pred, df["cluster"], average='macro')))
    print("F1 (Weighted): {}".format(f1_score(pred, df["cluster"], average='weighted')))

    #Visualize
    print_importances(rf.feature_importances_,features)
    visualize_random_forest(rf,features,"cluster","img/rf-class")

def df_random_forest_regression(df:pd.DataFrame, features, vars, train_df, test_df, max_leaf_nodes=None, n_trees=10, cluster_col="cluster") -> tuple:
    #Identify clusters of performance and take stratified random sample
    clusters = create_clusters(test_df, cluster_col=cluster_col)
    rf = RandomForestRegressor(n_estimators=n_trees, random_state=1, verbose=4, max_leaf_nodes=max_leaf_nodes)
    rf.fit(train_df[features],train_df[vars])
    score = rf.score(test_df[features],test_df[vars]))
    feature_importances = rf.feature_importances_
    return (score, feature_importances)

    #Visualize the importance of the features
    #print_importances(rf.feature_importances_,features)
    #visualize_random_forest(rf,features, cluster_col,"img/rf-reg")

def df_linreg(df:pd.DataFrame, features, vars, cluster_col="cluster"):
    #Linear regression
    feature_df = RobustScaler().fit_transform(df[features])
    reg = LinearRegression(fit_intercept=False).fit(feature_df, df[vars])
    score = reg.score(feature_df, df[vars])
    importances = reg.coef_[0]
    print("Score: {}".format(score))
    print_importances(np.absolute(importances),features)

def df_logistic_regression(df:pd.DataFrame, features, cluster_col="cluster"):
    #Logistic Regression
    feature_df = RobustScaler().fit_transform(df[features])
    model = LogisticRegression(multi_class='multinomial')
    model.fit(feature_df, df[cluster_col])
    score = model.score(feature_df, df[cluster_col])
    importances = model.coef_[0]

def df_xgboost_regression(df:pd.DataFrame, features, vars):
    return

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

#PCA on FEATURES
if case == 0:
    df_pca(df, features=FEATURES)

#KMeans on PERFORMANCE and basic stat each cluster
elif case == 1:
    df = df_kmeans(df, features=FEATURES, k=6)
    clusters = create_clusters(df)
    pp.pprint(analyze_clusters(clusters, features=PERFORMANCE))

#Guassian mixture on PERFORMANCE
elif case == 2:
    df = df_gauss_mixtures(df, features=PERFORMANCE, k=6)
    clusters = create_clusters(df)
    pp.pprint(analyze_clusters(clusters, features=PERFORMANCE))

#Agglomerative Clustering on PERFORMANCE
elif case == 3:
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

#Random Forest Regressor FEATURES -> PERFORMANCE
elif case == 10:
    df_random_forest_regression(df, features=FEATURES, vars=PERFORMANCE, max_leaf_nodes=256, n_trees=3, cluster_col="cluster_RUN_TIME")

#Random Forest Classifier FEATURES -> PERFORMANCE
elif case == 11:
    df_random_forest_classifier(df, features=FEATURES, vars=PERFORMANCE, max_leaf_nodes=8, k=8, cluster_col="cluster_RUN_TIME")

#Linear Regression with Features FEATURES -> PERFORMANCE
elif case == 12:
    df_linreg(df, features=FEATURES, vars=PERFORMANCE, cluster_col="cluster_RUN_TIME")

#Logistic Regression FEATURES -> PERFORMANCE
elif case == 13:
    df_logistic_regression(df, features=FEATURES, cluster_col="cluster_RUN_TIME")
