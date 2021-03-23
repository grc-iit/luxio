from luxio.emulator.generate_ior_trace_json_file import GenIORTraceJsonFile
from luxio.common.constants import *
import pandas as pd

import pprint, warnings

pp = pprint.PrettyPrinter(depth=6)

NPROCS = 16*40
GenIORTraceJsonFile().gen(NPROCS, nreads, nwrites, req_size, FPP, FSYNC_PER_WRITE, FSYNC, rand).save("")
