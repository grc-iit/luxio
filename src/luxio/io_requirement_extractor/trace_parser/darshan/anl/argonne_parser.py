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
        self.df = pd.read_csv(params['argonnne_all'])

    def standardize(self):
        """
        Preprocess the data as follows:
            Remove all entries where no I/O is occurring
            Create a score for total bytes
            Create a score for randomness
        """

        #Standardize the dataset
        super().standardize()
        
        df = self.df
        
        #Load numerical features
        numerical_features = self._minimum_features(__file__, "numerical.csv")
        #Remove negative counters
        df = df[(df[numerical_features] >= 0).all(axis=1)]
        #Select apps where at least 10 seconds of I/O occurred
        #df = df[df.TOTAL_IO_TIME > 10000]
        #Select apps where at least 15% of runtime is spent in I/O
        df['IO_FRAC'] = (df.TOTAL_IO_TIME/1000) / df.RUN_TIME
        #df = df[df.IO_FRAC > .15]
        #Fill NAs with 0s
        df = df.fillna(0)
        df.replace([np.inf, -np.inf], 0, inplace=True)

        #Fill categorical variables
        CATEGORICAL = ["MACHINE_NAME"]
        for categorial in CATEGORICAL:
            df["MACHINE_NAME_ID"] = 0
            df.loc[:, f"{categorial}_ID"] = pd.factorize(df[categorial])[0]

        self.df = df
        return df
