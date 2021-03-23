
from luxio.io_requirement_extractor.trace_parser import TraceParser
from typing import List, Dict, Tuple
from luxio.common.configuration_manager import *
from luxio.common.constants import *
import pandas as pd
import numpy as np
from functools import reduce

class SCSRedisParser(TraceParser):
    """
    A parser for Redis QoSA,
    empty since the SCSQosaParser uses redis format
    """
    def __init__(self) -> None:
        pass

    def _initialize(self):
        return

    def _finalize(self):
        return

    def parse(self, params) -> pd.DataFrame:
        """
        Parses the SCS stress test CSV and converts to pandas
        """
        #Load SCS QoSA CSV
        self._initialize()
        path = params["scs_redis_csv"]
        self.df = pd.read_csv(path)
        self._finalize()
        return self.df

    def standardize(self) -> None:
        df = self.df

        #Set storage name
        df.loc[:,'storage'] = "Redis"
        #Set interface
        df.loc[:,'interface'] = StorageInterface.KVS
        #Set capacity
        df.loc[df.device == 'hdd', 'capacity'] = 1*(TB/GB)*df[df.device == 'hdd']['servers'].to_numpy() #1TB / node
        df.loc[df.device == 'ssd', 'capacity'] = 250*(GB/GB)*df[df.device == 'ssd']['servers'].to_numpy() #250GB / node
        df.loc[df.device == 'nvme', 'capacity'] = 250*(GB/GB)*df[df.device == 'nvme']['servers'].to_numpy() #250GB / node
        #Set malleability
        df.loc[:,'malleable'] = 0

        return
