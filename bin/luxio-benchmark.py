
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

def naiive_algo(io_identifier:pd.DataFrame, app_classifier, storage_classifier, counter) -> pd.DataFrame:
    sslos = storage_classifier.sslos_
    #io_identifier = app_classifier.standardize(io_identifier)
    ranked_sslo = sslos.iloc[counter:,:]
    cost = emulated_cost(io_identifier, ranked_sslo)
    return cost

def optimal_algo(io_identifier:pd.DataFrame, app_classifier, storage_classifier, counter) -> pd.DataFrame:
    io_identifier = app_classifier.standardize(io_identifier)
    ranked_sslos = storage_classifier.get_coverages(io_identifier)
    ranked_sslos = ranked_sslos.nlargest(20, "magnitude")
    ranked_sslos.sort_values("magnitude")
    return emulated_cost(io_identifier, ranked_sslos)
    return 0

def luxio_algo(io_identifier:pd.DataFrame, app_classifier, storage_classifier, counter) -> pd.DataFrame:
    """
    Takes in an I/O identifier and produces a ranked list of candidate sslos to pass to the
    resource resolver.
    """
    #Get the fitness vector of the IOIdentifier to all of the classes
    fitnesses = app_classifier.get_fitnesses(io_identifier)
    #Multiply fitness and coverage
    ranked_sslos = app_classifier.sslos_.copy()
    ranked_sslos.loc[:,app_classifier.scores] = fitnesses[app_classifier.scores].to_numpy() * app_classifier.sslos_[app_classifier.scores].to_numpy()
    #Select the best 20 sslos
    ranked_sslos = ranked_sslos.groupby(["sslo_id"]).max().nlargest(20, "magnitude")
    #Sort the sslos in descending order
    ranked_sslos.sort_values("magnitude")
    cost = emulated_cost(io_identifier, ranked_sslos)
    return cost

def emulated_cost(io_identifier:pd.DataFrame, ranked_sslos:pd.DataFrame):
    MD = ["TOTAL_STDIO_OPENS", "TOTAL_POSIX_OPENS", "TOTAL_MPIIO_COLL_OPENS", "TOTAL_POSIX_STATS", "TOTAL_STDIO_SEEKS"]
    best_sslo = ranked_sslos.iloc[0,:]
    io_identifier = io_identifier.iloc[0,:]
    read_time = (io_identifier["TOTAL_BYTES_READ"] / (1<<20)) / best_sslo["Read_Large_BW"] if best_sslo["Read_Large_BW"] > 0 else 0
    write_time = (io_identifier["TOTAL_BYTES_WRITTEN"] / (1<<20)) / best_sslo["Write_Large_BW"] if best_sslo["Write_Large_BW"] > 0 else 0
    md_time = (io_identifier[MD].sum()*1024 / (1<<20)) / (best_sslo["Read_Small_BW"] + best_sslo["Write_Small_BW"])/2 if best_sslo["Read_Small_BW"] + best_sslo["Write_Small_BW"] > 0 else 0
    read_time /= io_identifier["NPROCS"]
    write_time /= io_identifier["NPROCS"]
    return (read_time + write_time + md_time), best_sslo["Price"]

def trials(algo, algo_name, df, ac, sc, records, n, top_n, rep=10):
    t = Timer()
    counter = 0
    for idx,row in df.iterrows():
        t.reset()
        for i in range(rep):
            t.resume()
            runtime,price = algo(row.to_frame().transpose(), ac, sc, counter)
            t.pause()
        counter += 1
        records.append({
            "APP": row["APP_NAME"],
            "ALGO": algo_name,
            "EST_RUNTIME": runtime,
            "PRICE": price,
            "MAP_THRPT": 1000 / (t.msec() / rep),
            "n" : n,
            "top_n" : top_n
        })

def sslo_gen(n=10**7):
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

    n_sslo = 100

    nPr = n_sslo

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
    df.sort_values(by="Read_Large_BW", inplace=True)
    bc = StorageClassifier(pd.DataFrame())
    bc.fit(df)
    return bc

if __name__ == "__main__":
    records = []
    df = pd.read_csv("datasets/df_subsample.csv")#.iloc[3:4,:]
    ac = AppClassifier.load("sample/app_classifier/app_class_model.pkl")
    ns = [10**3, 10**4, 10**5, 10**6, 10**7]
    ns = [10**3]
    top_ns = [10]
    for n in ns:
        for top_n in top_ns:
            sc = sslo_gen(n)
            print("sslos generated")
            ac.filter_sslos(sc, top_n=top_n)
            print("sslos filtered")
            trials(naiive_algo, "naiive_algo", df, ac, sc, records, n, top_n)
            trials(optimal_algo, "optimal_algo", df, ac, sc, records, n, top_n)
            trials(luxio_algo, "luxio_algo", df, ac, sc, records, n, top_n)
    pd.DataFrame(records).to_csv("metrics.csv")
