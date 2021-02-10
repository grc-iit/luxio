
from luxio.storage_requirement_builder.models import *
from typing import List, Dict, Tuple
import pandas as pd

from time import process_time_ns
from itertools import cycle, permutations, product

class Timer:
    def __init__(self):
        self.count = 0
        return

    def resume(self):
        self.start = process_time_ns()

    def pause(self):
        self.end = process_time_ns()
        self.count += self.end - self.start
        return self

    def reset(self):
        self.count = 0

    def msec(self):
        return self.count/10**6

def naiive_algo(io_identifier:pd.DataFrame, app_classifier, storage_classifier) -> pd.DataFrame:
    qosas = storage_classifier.qosas
    io_identifier = app_classifier.standardize(io_identifier)
    ranked_qosa = storage_classifier.get_coverages(io_identifier, qosas.iloc[naiive_algo.counter,:].to_frame().transpose())
    naiive_algo.counter += 1
    return emulated_cost(io_identifier, ranked_qosa)
naiive_algo.counter = 0

def optimal_algo(io_identifier:pd.DataFrame, app_classifier, storage_classifier) -> pd.DataFrame:
    io_identifier = app_classifier.standardize(io_identifier)
    ranked_qosas = storage_classifier.get_coverages(io_identifier)
    #ranked_qosas = ranked_qosas[ranked_qosas.magnitude > ranked_qosas.magnitude.quantile(.9)]
    ranked_qosas = ranked_qosas.nlargest(20, "magnitude")
    ranked_qosas.sort_values("magnitude")
    return emulated_cost(io_identifier, ranked_qosas)

def luxio_algo(io_identifier:pd.DataFrame, app_classifier, storage_classifier) -> pd.DataFrame:
    """
    Takes in an I/O identifier and produces a ranked list of candidate QoSAs to pass to the
    resource resolver.
    """
    #Get the fitness vector of the IOIdentifier to all of the classes
    fitnesses = app_classifier.get_fitnesses(io_identifier)
    #Multiply fitness and coverage
    ranked_qosas = app_classifier.qosas.copy()
    ranked_qosas.loc[:,app_classifier.scores] = fitnesses[app_classifier.scores].to_numpy() * app_classifier.qosas[app_classifier.scores].to_numpy()
    #Select the best 20 qosas
    ranked_qosas = ranked_qosas.groupby(["qosa_id"]).max().nlargest(20, "magnitude")
    #Sort the QoSAs in descending order
    ranked_qosas.sort_values("magnitude")
    return emulated_cost(io_identifier, ranked_qosas)

def emulated_cost(io_identifier:pd.DataFrame, ranked_qosas:pd.DataFrame):
    MD = ["TOTAL_STDIO_OPENS", "TOTAL_POSIX_OPENS", "TOTAL_MPIIO_COLL_OPENS", "TOTAL_POSIX_STATS", "TOTAL_STDIO_SEEKS"]
    best_qosa = ranked_qosas.iloc[0,:]
    io_identifier = io_identifier.iloc[0,:]
    read_time = (io_identifier["TOTAL_BYTES_READ"] / (1<<20)) / best_qosa["Read_Large_BW"] if best_qosa["Read_Large_BW"] > 0 else 0
    write_time = (io_identifier["TOTAL_BYTES_WRITTEN"] / (1<<20)) / best_qosa["Write_Large_BW"] if best_qosa["Write_Large_BW"] > 0 else 0
    md_time = (io_identifier[MD].sum()*1024 / (1<<20)) / (best_qosa["Read_Small_BW"] + best_qosa["Write_Small_BW"])/2 if best_qosa["Read_Small_BW"] + best_qosa["Write_Small_BW"] > 0 else 0
    read_time /= io_identifier["NPROCS"]
    write_time /= io_identifier["NPROCS"]
    return (read_time + write_time + md_time)

def trials(algo, algo_name, df, ac, sc, records, n=10):
    for idx,row in df.iterrows():
        t.reset()
        for i in range(n):
            t.resume()
            runtime = algo(row.to_frame().transpose(), ac, sc)
            t.pause()
        runtime /= n
        records.append({
            "APP": row["APP_NAME"],
            "ALGO": algo_name,
            "EST_RUNTIME": runtime,
            "MAP_TIME": t.msec()
        })

if __name__ == "__main__":
    t = Timer()
    records = []
    df = pd.read_csv("datasets/df_subsample.csv")

    ac = AppClassifier.load("sample/app_classifier/app_class_model.pkl")
    sc = StorageClassifier.load("sample/qosa_classifier/qosa_class_model.pkl")

    to_filter = False
    if to_filter == True:
        ac.filter_qosas(sc)
        ac.save("sample/app_classifier/app_class_model.pkl")
    #quit()

    trials(naiive_algo, "naiive_algo", df, ac, sc, records)
    trials(optimal_algo, "optimal_algo", df, ac, sc, records)
    trials(luxio_algo, "luxio_algo", df, ac, sc, records)
    pd.DataFrame(records).to_csv("metrics.csv")
