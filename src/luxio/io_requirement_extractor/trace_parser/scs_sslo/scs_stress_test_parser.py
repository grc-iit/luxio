
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

        UNIQUE = ['mpi_procs', 'clients', 'servers', 'network', 'storage', 'device', 'TroveSyncMeta', 'TroveSyncData',
                  'TroveMaxConcurrentIO', 'TCPBufferReceive', 'TCPBindSpecific', 'io_type', 'req_size', 'total_size_per_proc']
        MERGE = UNIQUE
        RSRC = ["hdd", "ssd", "nvme"]
        BWS = ["write_bw", "read_bw"]
        NEW_BWS = ["sequential_write_bw", "sequential_read_bw", "sequetial_mixed_bw", "random_write_bw", "random_read_bw", "random_mixed_bw"]
        INTERFERENCE = ["sequential_write_if", "random_if", "mixed_if"]

        #Create device type mapping
        MB = (1<<20)
        df = self.df
        df.loc[:,"storage_id"] = pd.factorize(df["storage"])[0]
        df.loc[:,"device_id"] = pd.factorize(df["device"])[0]
        df.loc[:,"io_type_id"] = pd.factorize(df["io_type"])[0]
        df = df.groupby(UNIQUE).mean().reset_index().fillna(0)

        #Create column per-device
        device_types = df['device'].unique()
        for device_type in device_types:
            df[device_type] = 0
            df.loc[df.device == device_type, f"{device_type}"] = df['servers']

        # Calculate seq, random, and mixed bw (MB/s) for different request sizes
        dfs = []
        for io_type in ['sequential', 'random', 'mixed-sequential', 'mixed-random']:
            sub_df = df[(df.io_type == io_type)]
            bws = {bw_type: f"{io_type}_{bw_type}_{req_size_id}" for bw_type in BWS}
            sub_df.rename(columns=bws, inplace=True)
            dfs.append(sub_df[MERGE + list(bws.values())])

        # Set interface
        df.loc[:, 'interface'] = StorageInterface.POSIX | StorageInterface.STDIO | StorageInterface.MPI
        # Set capacity
        df.loc[df.device == 'hdd', 'capacity'] = 1 * (TB / GB) * df[df.device == 'hdd']['servers'].to_numpy()  # 1TB / node
        df.loc[df.device == 'ssd', 'capacity'] = 250 * (GB / GB) * df[df.device == 'ssd']['servers'].to_numpy()  # 250GB / node
        df.loc[df.device == 'nvme', 'capacity'] = 250 * (GB / GB) * df[df.device == 'nvme']['servers'].to_numpy()  # 250GB / node
        # Set malleability
        df.loc[:, 'malleable'] = 0

        self.df = df
