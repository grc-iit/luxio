from luxio.io_requirement_extractor.trace_parser.darshan import DarshanTraceParser
from .theta_parser import ThetaTraceParser
from .mira_parser import MiraTraceParser
import pandas as pd

class ArgonneTraceParser(DarshanTraceParser):
    def __init__(self):
        self.theta = ThetaTraceParser()
        self.mira = MiraTraceParser()

    def _initialize(self):
        self.conf = ConfigurationManager.get_instance()

    def _finalize(self):
        return

    def parse(self):
        self.theta.parse()
        self.mira.parse()

    def standardize(self):
        """
        Preprocess the data as follows:
            Remove all entries where no I/O is occurring
            Create a score for total bytes
            Create a score for randomness
        """

        self.mira.standardize()
        self.theta.standardize()
        self.df = self.combine([self.mira.df, self.theta.df])
        super().standardize()
