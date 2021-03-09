from luxio.simulator.generate_ior_trace_json_file import GenIORTraceJsonFile
import pandas as pd

KB = 1<<10
MB = 1<<20
GB = 1<<20

gen = GenIORTraceJsonFile()
NPROCS = [1*40, 4*40, 8*40, 16*40]
REQ_SIZE = [1024, 1048576, 16777216]
SIZE_PER_PROC = [96*MB, 256*MB, 1*GB, 16*GB]
RFRAC = [0, .5, 1]
FPP = 1
FSYNC_PER_WRITE = 0
FSYNC = 0
RAND = [0, 1]

traces = []
for nprocs in NPROCS:
    for req_size in REQ_SIZE:
        for size_per_proc in SIZE_PER_PROC:
            for rfrac in RFRAC:
                for rand in RAND:
                    nreads = (size_per_proc/req_size)*nprocs*rfrac
                    nwrites = (size_per_proc/req_size)*nprocs*(1 - rfrac)
                    traces.append(gen.gen_IOR_trace(nprocs, nreads, nwrites, req_size, FPP, FSYNC_PER_WRITE, FSYNC, rand))

print(len(traces))
pd.DataFrame(traces).to_csv("sample.csv", index=False)
