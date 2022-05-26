
import os
import darshan
from luxio.io_requirement_extractor.trace_parser.darshan import DarshanTraceParser
from typing import List, Dict, Tuple
from luxio.common.configuration_manager import *
import pandas as pd
import numpy as np
import re

import pprint, warnings

pp = pprint.PrettyPrinter(depth=6)

class DarshanTraceParser_3_2_1(DarshanTraceParser):
    """
    A Darshan Parser to extract certain Variables for Luxio
    """
    def __init__(self) -> None:
        darshan.enable_experimental()
        pass

    def _initialize(self):
        return

    def _finalize(self):
        return

    def parse(self, params) -> pd.DataFrame:
        """
        Parses an inputted Darshan File and returns all Darshan variables
        """
        #Load Darshan features
        if 'dir' in params:
            dir = params['dir']
            paths = [ os.path.join(dir,filename) for filename in os.listdir(dir) if re.search(".darshan", filename)]
        elif 'path' in params:
            paths = [params['path']]

        features = {}
        self.reports_ = []
        for path in paths:
            print(path)
            report = darshan.DarshanReport(path, read_all=True)
            for io_interface, io_interface_features in report.agg_ioops().items():
                for feature,value in io_interface_features.items():
                    if re.search("FASTEST", feature) or \
                            re.search("_MAX_", feature) or re.search("TIMESTAMP", feature) or \
                            re.search("ALIGNMENT", feature):
                        if feature not in features:
                            features[feature] = 0
                        features[feature] = max(features[feature], value)
                    elif re.search("SLOWEST", feature) or re.search("_MIN_", feature):
                        if feature not in features:
                            features[feature] = np.inf
                        features[feature] = min(features[feature], value)
                    elif re.search("MODE", feature) or re.search("HINT", feature):
                        if feature not in features:
                            features[feature] = 0
                        features[feature] |= value
                    elif re.search("ALIGNMENT", feature):
                        if feature not in features:
                            features[feature] = 0
                        features[feature] = value
                    else:
                        if feature not in features:
                            features[feature] = 0
                        if isinstance(value, int) or isinstance(value, float):
                            features[feature] += value
                        else:
                            features[feature] = value

        #Convert features into dataframe
        min_features = self._minimum_features(__file__)
        self.df = pd.DataFrame(features, index=[0], columns=min_features)
        return self.df

    def standardize(self):
        """
        Converts the Darshan variables into variables that were used to train the luxio models
        """
        df = self.df

        #Prepend the word "TOTAL" to everything (model was trained on this kind of dataset)
        df = df.rename(columns={feature : f"TOTAL_{feature}" for feature in self.df.columns if feature != "NPROCS"})
        df = df.fillna(0)

        #Create the TOTAL_BYTES_READ
        df["TOTAL_BYTES_READ"] = (
                df["TOTAL_POSIX_BYTES_READ"] + df["TOTAL_STDIO_BYTES_READ"] + df["TOTAL_MPIIO_BYTES_READ"]
        )
        df["TOTAL_BYTES_WRITTEN"] = (
                df["TOTAL_POSIX_BYTES_WRITTEN"] + df["TOTAL_STDIO_BYTES_WRITTEN"] + df["TOTAL_MPIIO_BYTES_WRITTEN"]
        )

        #Ignore negative timing measurements
        times = [
            "TOTAL_POSIX_F_READ_TIME",
            "TOTAL_MPIIO_F_READ_TIME",
            "TOTAL_POSIX_F_WRITE_TIME",
            "TOTAL_MPIIO_F_WRITE_TIME",
            "TOTAL_POSIX_F_META_TIME",
            "TOTAL_MPIIO_F_META_TIME"
        ]
        for time in times:
            df.loc[df[time] < 0,time] = 0

        self.df = df.fillna(0)

        #Create derived columns
        super().standardize()
