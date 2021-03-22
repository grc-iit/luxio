
from luxio.io_requirement_extractor.trace_parser import TraceParser
from typing import List, Dict, Tuple
from luxio.common.configuration_manager import *
import pandas as pd
import numpy as np
from functools import reduce

from .redis import SCSRedisParser
from .orangefs import SCSOrangeFSParser

class SCSStressTestParser(TraceParser):
    """
    A Darshan Parser to extract certain Variables for Luxio
    """
    def _initialize(self):
        self.conf = ConfigurationManager.get_instance()
        self.redis = SCSRedisParser()
        self.orangefs = SCSOrangeFSParser()

    def _finalize(self):
        return

    def parse(self) -> pd.DataFrame:
        """
        Parses the SCS stress test CSV and converts to pandas
        """
        self.redis.parse()
        self.orangefs.parse()

    def standardize(self):
        """
        Converts the SCS stress test into the standard format used to train Luxio models
        """

        UNIQUE = ["clients", "servers", "network", "device", "config", "io_type", "req_size", "storage"]

        #Standardize datasets
        self.redis.standardize()
        self.orangefs.standardize()
        self.df = self.combine([self.redis.df, self.orangefs.df])

        #Create device type mapping
        MB = (1<<20)
        df = self.df
        df.loc[:,"storage_id"] = pd.factorize(df["storage"])[0]
        df.loc[:,"device_id"] = pd.factorize(df["device"])[0]
        df.loc[:,"io_type_id"] = pd.factorize(df["io_type"])[0]
        df.loc[:,"read_bw"] = (df['total_size_per_proc']/(2*MB))*df['clients']*40 / df['read_time']
        df.loc[:,"write_bw"] = (df['total_size_per_proc']/(2*MB))*df['clients']*40 / df['write_time']
        df.loc[:,"mdm_thrpt"] = (df['mdm_reqs_per_proc'] / df['mdm_time'])
        df = df.groupby(UNIQUE).mean().reset_index().fillna(0)

        #Create compact resource vector
        #device_types = list(df['device'])
        #device_counts = list(df['servers'])
        #resources = [{type:count} for type,count in zip(device_types, device_counts)]
        #df.loc[:,'resources'] = pd.Series(resources)

        #Create column per-device
        device_types = df['device'].unique()
        for device_type in device_types:
            df[device_type] = 0
            df.loc[df.device == device_type, f"{device_type}"] = df['servers']

        self.df = df
