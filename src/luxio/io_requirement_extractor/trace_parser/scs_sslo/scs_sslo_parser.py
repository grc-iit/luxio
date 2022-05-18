
from luxio.io_requirement_extractor.trace_parser import TraceParser
from typing import List, Dict, Tuple
from luxio.common.configuration_manager import *
from luxio.common.constants import *
import pandas as pd
import numpy as np
from functools import reduce

from .redis import SCSRedisParser
from .orangefs import SCSOrangeFSParser

class SCSSSLOParser(TraceParser):
    """
    A Darshan Parser to extract certain Variables for Luxio
    """
    def _initialize(self):
        return

    def _finalize(self):
        return

    def parse(self, params) -> pd.DataFrame:
        """
        Parses the SCS stress test CSV and converts to pandas
        """
        path = params["stress_tests"]
        self.df = pd.read_csv(path)
        return self.df

    def standardize(self):
        """
        Converts the SCS stress test into the standard format used to train Luxio models
        """

        MERGE = ["clients", "servers", "network", "device", "device_id",
                 "config", "TroveSyncMeta", "TroveSyncMeta_id",
                 "TroveSyncData", "TroveSyncData_id",
                 "TroveMaxConcurrentIO", "TCPBufferReceive", "TCPBindSpecific_id",
                 "storage", "storage_id", "capacity", "malleable", "interface"]
        RSRC = ["hdd", "ssd", "nvme"]
        BWS = ["write_bw", "read_bw"]
        THRPTS = []
        SEQ = [
            #"sequential_write_bw_small", "sequential_read_bw_small",
            "sequential_write_bw_large", "sequential_read_bw_large"
        ]
        RAND = [
            #"random_write_bw_small", "random_read_bw_small",
            "random_write_bw_large", "random_read_bw_large"
        ]

        df = self.df

        #Classify request sizes
        df['req_size_id'] = "small"
        df.loc[df.req_size <= 64, 'req_size_id'] = "small"
        df.loc[df.req_size > 64, 'req_size_id'] = "large"
        #req_size_ids = ["small","large"]
        req_size_ids = ["small","large"]
        io_types = ["random", "sequential", "mixed-sequential", "mixed-random"]

        #Calculate seq and random bw (MB/s) for different request sizes
        dfs = []
        for req_size_id in req_size_ids:
            for io_type in io_types:
                sub_df = df[(df.io_type == io_type) & (df.req_size_id == req_size_id)]
                if len(sub_df) == 0:
                    continue
                bws = {bw_type:f"{io_type}_{bw_type}_{req_size_id}" for bw_type in BWS}
                sub_df.rename(columns=bws, inplace=True)
                dfs.append(sub_df[MERGE + list(bws.values())])

        #Add medata throughput
        #dfs.append(df[MERGE + THRPTS + RSRC])

        #Combine the partial dataframes
        df = reduce(lambda x, y: pd.merge(x, y, on=MERGE, how='outer'), dfs)
        df = df.fillna(0)

        #Emulate mixedness
        req_size_ids = ["large"]
        io_types = ["mixed-sequential", "mixed-random"]
        bws = [f"{io_type}_{bw_type}_{req_size_id}" for io_type in io_types for req_size_id in req_size_ids for bw_type in BWS]
        tmpfs = df[df.device == 'tmpfs']
        client_scales = np.array(tmpfs['clients'].drop_duplicates())
        server_scales = np.array(tmpfs['servers'].drop_duplicates())
        for client_scale in client_scales:
            for server_scale in server_scales:
                idx = (tmpfs.clients == client_scale) & (tmpfs.servers == server_scale)
                tmpfs.loc[idx,bws] = tmpfs.loc[idx, bws].max().to_numpy()
        df.loc[df.device == 'tmpfs',:] = tmpfs

        #Sensitivity to Concurrency (ratio of std to mean)
        CONC = [feature for feature in MERGE if feature != 'clients']
        grp = df.groupby(CONC)
        df = grp.max().reset_index()
        df.loc[:,'sensitivity2concurrency'] = (grp[SEQ].std().to_numpy() / grp[SEQ].mean().to_numpy()).mean(axis=1)
        df = df.drop(columns='clients')
        df = df.fillna(0)

        #Sensitivity to Randomness (% faster seq is over rand)
        #Ideally is 0
        #> 0 -> randomness decreases performance
        #Otherwise, randomness increases performance
        df.loc[:,'sensitivity2randomness'] = (df[SEQ].to_numpy()/df[RAND].to_numpy() - 1).mean(axis=1)
        df.loc[:,'sequentiality'] = -1*df['sensitivity2randomness']
        df = df.fillna(0)

        #Deployment ID
        df.loc[:,'deployment_id'] = df.index

        self.df = df
