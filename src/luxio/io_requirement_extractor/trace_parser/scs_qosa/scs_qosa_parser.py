
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

        MD = ["mdm_reqs_per_proc"]
        DATA = ["total_r_or_w_size", "total_r_or_w_size"]
        RW_TIMES = ["write_time", "read_time"]
        MD_TIMES = ["mdm_time"]
        BWS = ["write_bw", "read_bw"]
        THRPT = ["mdm_thrpt"]
        MERGE = ["clients", "servers", "network", "device", "storage", "config"]

        #Get the total amount of data/md ops
        df.loc[:,'total_r_or_w_size'] = (df['total_size_per_proc']/2)*df['clients']*40
        df.loc[:,'total_mdm_reqs'] = df['mdm_reqs_per_proc']*df['clients']*40

        #Get the unique request sizes and I/O patterns
        req_size_ids = ['small', 'medium', 'large']
        req_sizes = np.sort(df['req_size'].unique())
        io_types = df['io_type'].unique()
        if len(req_sizes) != 3:
            raise
        for req_size, req_size_id in zip(req_sizes, req_size_ids):
            df.loc[df['req_size'] == req_size, 'req_size_id'] = req_size_id

        #Calculate seq and random bw (MB/s) for different request sizes
        dfs = []
        for req_size, req_size_id in zip(req_sizes, req_size_ids):
            for io_type in io_types:
                sub_df = df[(df.io_type == io_type) & (df.req_size == req_size)].drop(['req_size_id'], axis=1)
                sub_df.loc[:,BWS] = np.divide(sub_df[DATA].to_numpy()/(1<<20), sub_df[RW_TIMES].to_numpy(), out=np.zeros_like(sub_df[RW_TIMES]), where=sub_df[RW_TIMES]!=0)
                times = {col:f"{io_type}_{col}_{req_size_id}" for col in BWS}
                sub_df.rename(columns=times, inplace=True)
                dfs.append(sub_df[MERGE + list(times.values())])
        #Calculate md throughput for test cases
        grps = df.groupby(MERGE)
        sub_df = grps.max().reset_index()
        sub_df[THRPT] = np.divide(sub_df[MD].to_numpy(), sub_df[MD_TIMES].to_numpy(), out=np.zeros_like(sub_df[MD_TIMES]), where=sub_df[MD_TIMES]!=0)
        sub_df = sub_df[MERGE + THRPT]
        dfs.append(sub_df)
        #Calculate total time
        sub_df = grps.sum().reset_index()
        sub_df = sub_df[MERGE + RW_TIMES + MD_TIMES]
        dfs.append(sub_df)
        #Combine the partial dataframes
        df = reduce(lambda x, y: pd.merge(x, y, on=MERGE, how='outer'), dfs)

        #Make all elements numerical
        #df['device'].unqiue()

        self.df = df
