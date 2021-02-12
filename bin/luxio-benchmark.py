
from luxio.storage_requirement_builder.models import *
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np

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
    #io_identifier = app_classifier.standardize(io_identifier)
    ranked_qosa = qosas.iloc[naiive_algo.counter,:]
    naiive_algo.counter += 1
    #return emulated_cost(io_identifier, ranked_qosa)
    return 0
naiive_algo.counter = 0

def optimal_algo(io_identifier:pd.DataFrame, app_classifier, storage_classifier) -> pd.DataFrame:
    io_identifier = app_classifier.standardize(io_identifier)
    ranked_qosas = storage_classifier.get_coverages(io_identifier)
    #ranked_qosas = ranked_qosas[ranked_qosas.magnitude > ranked_qosas.magnitude.quantile(.9)]
    ranked_qosas = ranked_qosas.nlargest(20, "magnitude")
    ranked_qosas.sort_values("magnitude")
    #return emulated_cost(io_identifier, ranked_qosas)
    return 0

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
    #return emulated_cost(io_identifier, ranked_qosas)
    return 0

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

def trials(algo, algo_name, df, ac, sc, records, n, rep=10):
    t = Timer()
    for idx,row in df.iterrows():
        t.reset()
        for i in range(rep):
            t.resume()
            runtime = algo(row.to_frame().transpose(), ac, sc)
            t.pause()
        records.append({
            #"APP": row["APP_NAME"],
            "ALGO": algo_name,
            #"EST_RUNTIME": runtime,
            "MAP_THRPT": 1000 / (t.msec() / rep),
            "n" : n
        })

def qosa_gen(n=10**7):
    bandwidth_large = {
        'nvme': 10000,
        'ssd': 1000,
        'hdd': 100
    }

    bandwidth_small = {
        'nvme': 1000,
        'ssd': 100,
        'hdd': 10
    }

    cost = {
        'nvme': 1000,
        'ssd': 10,
        'hdd': 1

    }

    n_qosa = 100

    nPr = n_qosa

    #500*nvme + 100*ssd + hdd
    r = 3
    nvme = 2000
    ssd = 500
    hdd = 120
    count = int(np.ceil(n ** (1/3)))
    print(count)

    dev = filter(lambda x: x[0] != 0 or x[1] != 0 or x[2] != 0, product(*[range(count), range(count), range(count)]))

    price = []
    read_lbw = []
    write_lbw = []
    read_sbw = []
    write_sbw = []
    devv = []
    for i in dev:
        rlbw = (i[0] * bandwidth_large['nvme']) + (i[1] * bandwidth_large['ssd']) + (i[2] * bandwidth_large['hdd'])
        wlbw = (i[0] * bandwidth_large['nvme']) + (i[1] * bandwidth_large['ssd']) + (i[2] * bandwidth_large['hdd'])
        rsbw = (i[0] * bandwidth_small['nvme']) + (i[1] * bandwidth_small['ssd']) + (i[2] * bandwidth_small['hdd'])
        wsbw = (i[0] * bandwidth_small['nvme']) + (i[1] * bandwidth_small['ssd']) + (i[2] * bandwidth_small['hdd'])
        p = (i[0] * cost['nvme']) + (i[1] * cost['ssd']) + (i[2] * cost['hdd'])

        devv.append(i)
        price.append(p)
        read_lbw.append(rlbw)
        write_lbw.append(wlbw)
        read_sbw.append(rsbw)
        write_sbw.append(wsbw)

    df = pd.DataFrame(data=list(zip(devv, price, read_lbw, write_lbw, read_sbw, write_sbw)), columns=['Device','Price', 'Read_Large_BW', 'Write_Large_BW', 'Read_Small_BW', 'Write_Small_BW'])
    bc = StorageClassifier(pd.DataFrame())
    bc.fit(df)
    return bc

if __name__ == "__main__":
    records = []
    df = pd.read_csv("datasets/df_subsample.csv").iloc[0:1,:]
    ac = AppClassifier.load("sample/app_classifier/app_class_model.pkl")
    for n in [10**3, 10**4, 10**5, 10**6, 10**7]:
        print(n)
        sc = qosa_gen(n)
        print("Qosas generated")
        ac.filter_qosas(sc)
        print("Qosas filtered")
        trials(naiive_algo, "naiive_algo", df, ac, sc, records, n)
        trials(optimal_algo, "optimal_algo", df, ac, sc, records, n)
        trials(luxio_algo, "luxio_algo", df, ac, sc, records, n)
    pd.DataFrame(records).to_csv("metrics.csv")
