from luxio.emulator.generate_ior_trace_json_file import GenIORTraceJsonFile
from luxio.common.constants import *
import pandas as pd

import pprint, warnings

NCHECKS = 10
IO_INTENSITY = [.1, .25, .5, .75, .9]
NAME = ["compute_intense", "compute_heavy", "balanced", "data_heavy", "data_intense"]
RUNTIME = [200, 210, 220, 230, 240]
NPROCS = 16*40
REQ_SIZE = [1*KB, 1*MB, 4*MB, 16*MB, 32*MB]
RFRAC = 0
FPP = 1
FSYNC_PER_WRITE = 0
FSYNC = 0
RAND = 0

#"""
for name,intensity,req_size,runtime in zip(NAME, IO_INTENSITY, REQ_SIZE, RUNTIME):
    GenIORTraceJsonFile(
        read_bw_small=50,
        read_bw_med=50,
        read_bw_large=50,
        write_bw_small=50,
        write_bw_med=50,
        write_bw_large=50,
        mdm_thrpt = 50
    ).fixed_compute(5, 10, intensity, RFRAC, 16*MB, NPROCS, FPP, FSYNC_PER_WRITE, FSYNC, RAND).to_json(f"{name}.json")
#"""

for name,intensity,req_size,runtime in zip(NAME, IO_INTENSITY, REQ_SIZE, RUNTIME):
    """
    GenIORTraceJsonFile().cap_runtime(NCHECKS, intensity, RFRAC, req_size, runtime, NPROCS, FPP, FSYNC_PER_WRITE, FSYNC, RAND).to_json(f"{name}.json")
    """

    """
    GenIORTraceJsonFile(
        read_bw_small=50,
        read_bw_med=50,
        read_bw_large=50,
        write_bw_small=50,
        write_bw_med=50,
        write_bw_large=50,
        mdm_thrpt = 50
    ).cap_runtime(NCHECKS, intensity, RFRAC, req_size, runtime, NPROCS, FPP, FSYNC_PER_WRITE, FSYNC, RAND).to_json(f"{name}.json")
    """
