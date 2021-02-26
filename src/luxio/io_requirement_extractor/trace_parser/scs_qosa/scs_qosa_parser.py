
import darshan
from luxio.io_requirement_extractor.trace_parser import TraceParser
from typing import List, Dict, Tuple
from luxio.common.configuration_manager import *
import pandas as pd
import numpy as np
from functools import reduce

class SCSQosaParser(TraceParser):
    """
    A Darshan Parser to extract certain Variables for Luxio
    """
    def __init__(self) -> None:
        darshan.enable_experimental()
        pass

    def _initialize(self):
        self.conf = ConfigurationManager.get_instance()

    def _finalize(self):
        return

    def parse(self) -> pd.DataFrame:
        """
        Parses the SCS stress test CSV and converts to pandas
        """
        #Load SCS QoSA CSV
        conf = self.conf
        path = conf.scs_qosa_csv
        self.df = pd.read_csv(path)
        return self.df

    def standardize(self):
        """
        Converts the SCS stress test into the standard format used to train Luxio models
        """
        df = self.df

        DATA = ["mdm_reqs_per_proc", "total_rw_size", "total_rw_size"]
        TIMES = ["mdm_time", "write_time", "read_time"]
        BWS = ["mdm_thrpt", "write_bw", "read_bw"]

        #Set index to seq
        df.set_index('seq', inplace=True)

        #Get the total amount of data/md ops
        df.loc[:,'total_rw_size'] = (df['total_size_per_proc']/2)*df['clients']
        df.loc[:,'total_mdm_reqs'] = df['mdm_reqs_per_proc']*df['clients']

        #Get the unique request sizes and I/O patterns
        req_sizes = np.sort(df['req_size'].unique())
        io_types = df['io_type'].unique()
        if len(req_sizes) != 2:
            raise

        #Calculate seq and random bw for different request sizes
        dfs = []
        for io_type in io_types:
            for req_size in req_sizes:
                sub_df = df[(df.io_type == io_type) & (df.req_size == req_size)].drop(['io_type', 'req_size'], axis=1)
                sub_df.loc[:,BWS] = sub_df[DATA] / sub_df[TIMES]
                sub_df.rename(columns={col:f"{io_type}_{col}_{req_size}" for col in TIMES + BWS}, inplace=True)
                dfs.append(sub_df)
        df = reduce(lambda x, y: pd.merge(x, y), dfs)
        
        self.df = df
