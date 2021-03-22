
from luxio.io_requirement_extractor.trace_parser import TraceParser
from typing import List, Dict, Tuple
from luxio.common.configuration_manager import *
from luxio.common.constants import *
import pandas as pd
import numpy as np
from functools import reduce

class SCSOrangeFSParser(TraceParser):
    """
    A Darshan Parser to extract certain Variables for Luxio
    """

    def _initialize(self):
        self.conf = ConfigurationManager.get_instance()

    def _finalize(self):
        return

    def parse(self) -> pd.DataFrame:
        """
        Parses the SCS stress test CSV and converts to pandas
        """
        #Load SCS QoSA CSV
        self._initialize()
        conf = self.conf
        path = conf.scs_orangefs_csv
        self.df = pd.read_csv(path)
        self._finalize()
        return self.df

    def standardize(self) -> None:
        """
        Converts the SCS stress test into the standard format used to train Luxio models
        """
        df = self.df

        #Convert req_size to an integer number bytes
        df['suffix1'] = df['req_size'].str.extract(r'[0-9]+([km])')
        df['prefix1'] = df['req_size'].str.extract(r'([0-9]+)[km]')
        df['prefix1'] = df['prefix1'].astype(int)
        df.loc[df['suffix1'] == 'm','prefix_suffix'] = df['prefix1']*(1<<20)
        df.loc[df['suffix1'] == 'k','prefix_suffix'] = df['prefix1']*(1<<10)
        df['req_size'] = df['prefix_suffix']

        #Convert total_size_per_proc to an integer number bytes
        df['suffix1'] = df['total_size_per_proc'].str.extract(r'[0-9]+([km])')
        df['prefix1'] = df['total_size_per_proc'].str.extract(r'([0-9]+)[km]')
        df['prefix1'] = df['prefix1'].astype(int)
        df.loc[df['suffix1'] == 'm','prefix_suffix'] = df['prefix1']*(1<<20)
        df.loc[df['suffix1'] == 'k','prefix_suffix'] = df['prefix1']*(1<<10)
        df['total_size_per_proc'] = df['prefix_suffix']

        #Get read_time and write_time
        MB = (1<<20)
        df['read_time'] = (df['total_size_per_proc']/(2*MB))*df['clients']*40 / df['read_bw']
        df['write_time'] = (df['total_size_per_proc']/(2*MB))*df['clients']*40 / df['write_bw']

        #Convert config to an integer
        df['config'] = df['config'].str.extract(r'[a-zA-Z]+_([0-9]+)').astype(int) + 100

        #Set storage name
        df.loc[:,'storage'] = "OrangeFS"
        #Set interface
        df.loc[:,'interface'] = StorageInterface.POSIX | StorageInterface.STDIO | StorageInterface.MPI
        #Set capacity
        df.loc[df.device == 'hdd', 'capacity'] = 1*(TB/GB)*df[df.device == 'hdd']['servers'].to_numpy() #1TB / node
        df.loc[df.device == 'ssd', 'capacity'] = 250*(GB/GB)*df[df.device == 'ssd']['servers'].to_numpy() #250GB / node
        df.loc[df.device == 'nvme', 'capacity'] = 250*(GB/GB)*df[df.device == 'nvme']['servers'].to_numpy() #250GB / node
        #Set malleability
        df.loc[:,'malleable'] = 0

        self.df = df
