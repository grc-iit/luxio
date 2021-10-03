
from luxio.io_requirement_extractor.trace_parser import TraceParser
from typing import List, Dict, Tuple
from luxio.common.configuration_manager import *
from luxio.common.constants import *
import pandas as pd
import numpy as np
from functools import reduce

from .redis import SCSRedisParser
from .orangefs import SCSOrangeFSParser

class SCSssloParser(TraceParser):
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
        path = params["scs_stress_test_csv"]
        self.df = pd.read_csv(path)
        return self.df

    def standardize(self):
        """
        Converts the SCS stress test into the standard format used to train Luxio models
        """

        MD = ["mdm_reqs_per_proc"]
        DATA = ["total_r_or_w_size", "total_r_or_w_size"]
        TIMES = ["write_time", "read_time", "mdm_time"]
        MERGE = ["clients", "servers", "network", "device", "config", "storage", "capacity", "malleable", "interface", "storage_id", "device_id"]
        RSRC = ["hdd", "ssd", "nvme"]
        BWS = ["write_bw", "read_bw"]
        THRPTS = ["mdm_thrpt"]
        SEQ = [
            "sequential_write_bw_small", "sequential_read_bw_small",
            "sequential_write_bw_large", "sequential_read_bw_large"
        ]
        RAND = [
            "random_write_bw_small", "random_read_bw_small",
            "random_write_bw_large", "random_read_bw_large"
        ]

        df = self.df

        #Classify request sizes
        df['req_size_id'] = "small"
        df.loc[df.req_size <= 64*KB, 'req_size_id'] = "small"
        df.loc[df.req_size > 64*KB, 'req_size_id'] = "large"
        req_size_ids = ["small","large"]
        io_types = ["random", "sequential"]

        #Calculate seq and random bw (MB/s) for different request sizes
        dfs = []
        for req_size_id in req_size_ids:
            for io_type in io_types:
                sub_df = df[(df.io_type == io_type) & (df.req_size_id == req_size_id)]
                bws = {bw_type:f"{io_type}_{bw_type}_{req_size_id}" for bw_type in BWS}
                sub_df.rename(columns=bws, inplace=True)
                dfs.append(sub_df[MERGE + list(bws.values())])

        #Add medata throughput
        dfs.append(df[MERGE + THRPTS + RSRC])

        #Combine the partial dataframes
        df = reduce(lambda x, y: pd.merge(x, y, on=MERGE, how='outer'), dfs)

        #Sensitivity to Concurrency (% )
        CONC = ['network', 'device', 'config', 'storage', 'capacity']
        grp = df.groupby(CONC)
        df = grp.max().reset_index()
        df.loc[:,'sensitivity2concurrency'] = (grp[SEQ+RAND].std().to_numpy() / grp[SEQ+RAND].mean().to_numpy()).mean(axis=1)
        df = df.drop(columns='clients')

        #Sensitivity to Randomness (% faster seq is over rand)
        df.loc[:,'sensitivity2randomness'] = (df[SEQ].to_numpy()/df[RAND].to_numpy() - 1).mean(axis=1)
        df.loc[:,'sequentiality'] = -1*df['sensitivity2randomness']

        #Deployment ID
        df.loc[:,'deployment_id'] = df.index

        self.df = df
