
from luxio.io_requirement_extractor.trace_parser import TraceParser
from typing import List, Dict, Tuple
from luxio.common.configuration_manager import *
from luxio.common.constants import *
import pandas as pd
import numpy as np
from functools import reduce

class SCSStressTestParser(TraceParser):
    """
    A Darshan Parser to extract certain Variables for Luxio
    """
    def _initialize(self):
        self.conf = ConfigurationManager.get_instance()

    def _finalize(self):
        return

    def parse(self, params) -> pd.DataFrame:
        self._initialize()
        path = params["stress_tests_csv"]
        self.df = pd.read_csv(path)
        self._finalize()
        return self.df

    def standardize(self):
        """
        Converts the SCS stress test into the standard format used to train Luxio models
        """

        #Convert categorical to numerical
        CATEGORICAL = [
            'device',
            'storage',
            'TroveSyncMeta',
            'TroveSyncData',
            'TCPBufferReceive',
            'TCPBindSpecific',
            'io_type'
        ]
        df = self.df
        for categorial in CATEGORICAL:
            df.loc[:, f"{categorial}_id"] = pd.factorize(df[categorial])[0]

        # Set interface
        df.loc[:, 'interface'] = StorageInterface.POSIX | StorageInterface.STDIO | StorageInterface.MPI

        # Set capacity
        df.loc[df.device == 'tmpfs', 'capacity'] = 32 * (GB / GB) * df[df.device == 'tmpfs']['servers'].to_numpy()  # 32GB / node
        df.loc[df.device == 'ssd', 'capacity'] = 250 * (GB / GB) * df[df.device == 'ssd']['servers'].to_numpy()  # 250GB / node
        df.loc[df.device == 'nvme', 'capacity'] = 250 * (GB / GB) * df[df.device == 'nvme']['servers'].to_numpy()  # 250GB / node

        # Set malleability
        df.loc[:, 'malleable'] = 0

        self.df = df
