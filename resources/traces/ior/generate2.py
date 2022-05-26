from luxio.emulator.generate_ior_trace_json_file import GenIORTraceJsonFile
from luxio.common.constants import *
import pandas as pd

import pprint, warnings

pp = pprint.PrettyPrinter(depth=6)

NCHECKS = [1, 10]
NPROCS = [1*40, 16*40]
REQ_SIZE = [1*KB, 1*MB, 16*MB]
NREQS = [100, 1000]
RFRAC = [0, .5, 1]
FPP = [0,1]
FSYNC_PER_WRITE = 0
FSYNC = 0
RAND = 0

traces = []
for nchecks in NCHECKS:
    for nprocs in NPROCS:
        for req_size in REQ_SIZE:
            for nreqs in NREQS:
                for rfrac in RFRAC:
                    for fpp in FPP:
                        nreads = rfrac * nreqs
                        nwrites = (1-rfrac) * nreqs
                        traces.append(GenIORTraceJsonFile().gen(nchecks, 20, nprocs, nreads, nwrites, req_size, fpp, FSYNC_PER_WRITE, FSYNC, RAND).get())

pd.DataFrame(traces).to_csv("sample.csv", index=False)
