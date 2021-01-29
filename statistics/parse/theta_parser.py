from .generic_trace_parser import GenericTraceParser
import pandas as pd

class ThetaParser(GenericTraceParser):
    def __init__(self, path):
        self.path = path
        self.df = pd.read_csv(path)

    def clean(self):
        return

    def standardize(self):
        self.df.loc[:,"TOTAL_BYTES_READ"] = self.df.TOTAL_POSIX_BYTES_READ + self.df.TOTAL_STDIO_BYTES_READ + self.df.TOTAL_MPIIO_BYTES_READ
        self.df.loc[:,"TOTAL_BYTES_WRITTEN"] = self.df.TOTAL_POSIX_BYTES_WRITTEN + self.df.TOTAL_STDIO_BYTES_WRITTEN + self.df.TOTAL_MPIIO_BYTES_WRITTEN
