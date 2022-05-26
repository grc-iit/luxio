
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

        OFS_CONFIG = ['TroveSyncMeta', 'TroveSyncData', 'TroveMaxConcurrentIO', 'TCPBufferReceive', 'TCPBindSpecific']
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

        #Set orangefs config
        df.loc[:, 'config'] = df.groupby(OFS_CONFIG).ngroup()

        #Convert categorical to numerical
        for categorial in CATEGORICAL:
            df.loc[:, f"{categorial}_id"] = pd.factorize(df[categorial])[0]

        #Set interface
        df.loc[:, 'interface'] = StorageInterface.POSIX | StorageInterface.STDIO | StorageInterface.MPI

        #Emulate write BW
        df.loc[df.device == 'nvme', 'write_bw'] = df[df.device == 'nvme']['read_bw']*.7
        #df.loc[df.device == 'ssd', 'write_bw'] = df[df.device == 'ssd']['read_bw']*.7
        df = df[df.device != 'ssd']
        sub_df = df[df.device == 'nvme']
        sub_df['device'] = 'ssd'
        sub_df['write_bw'] *= .8
        sub_df['read_bw'] *= .8
        df = pd.concat([df, sub_df])

        # Set capacity
        df.loc[df.device == 'tmpfs', 'capacity'] = 32 * (GB / GB) * df[df.device == 'tmpfs']['servers'].to_numpy()  # 32GB / node
        df.loc[df.device == 'ssd', 'capacity'] = 250 * (GB / GB) * df[df.device == 'ssd']['servers'].to_numpy()  # 250GB / node
        df.loc[df.device == 'nvme', 'capacity'] = 250 * (GB / GB) * df[df.device == 'nvme']['servers'].to_numpy()  # 250GB / node

        # Set malleability
        df.loc[:, 'malleable'] = 0

        #Determine sequentiality of deployment
        df['sequential'] = 0
        df.loc[(df.io_type == 'sequential') | (df.io_type == 'mixed-sequential'), 'sequential'] = 1

        #Determine the mixedness of deployment
        df['mixed'] = 0
        df.loc[(df.io_type == 'mixed-sequential') | (df.io_type == 'mixed-random'), 'mixed'] = 1

        #Determine performance ratio of clients to servers
        df.loc[:,'interference'] = (df.read_bw + df.write_bw) / (2 * df.clients / df.servers)

        self.df = df
