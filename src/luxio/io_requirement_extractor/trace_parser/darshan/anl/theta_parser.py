from luxio.common.configuration_manager import *
from luxio.io_requirement_extractor.trace_parser.darshan import DarshanTraceParser
import pandas as pd

class ThetaTraceParser(DarshanTraceParser):
    def __init__(self):
        return

    def _initialize(self):
        return

    def _finalize(self):
        return

    def parse(self, params):
        self._initialize()
        self.df = pd.read_csv(params["theta_path"])
        self._finalize()

    def standardize(self):
        self.df.loc[:,"TOTAL_BYTES_READ"] = self.df.TOTAL_POSIX_BYTES_READ + self.df.TOTAL_STDIO_BYTES_READ + self.df.TOTAL_MPIIO_BYTES_READ
        self.df.loc[:,"TOTAL_BYTES_WRITTEN"] = self.df.TOTAL_POSIX_BYTES_WRITTEN + self.df.TOTAL_STDIO_BYTES_WRITTEN + self.df.TOTAL_MPIIO_BYTES_WRITTEN
