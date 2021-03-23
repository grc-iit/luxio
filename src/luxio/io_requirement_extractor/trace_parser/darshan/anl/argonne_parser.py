from luxio.common.configuration_manager import *
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

    def parse(self, params):
        self.theta.parse(params)
        self.mira.parse(params)

    def standardize(self):
        """
        Preprocess the data as follows:
            Remove all entries where no I/O is occurring
            Create a score for total bytes
            Create a score for randomness
        """

        #Standardize the dataset
        self.mira.standardize()
        self.theta.standardize()
        self.df = self.combine([self.mira.df, self.theta.df])
        super().standardize()

        #Load numerical features
        numerical_features = self._minimum_features(__file__, "numerical.csv")
        #Remove negative entries
        self.df = self.df[(self.df[numerical_features] >= 0).all(axis=1)]
        #Select only nonnegative I/O times
        self.df = self.df[self.df.TOTAL_IO_TIME > 0]
        #Fill NAs with 0s
        self.df = self.df.fillna(0)
