
from luxio.common.error_codes import ErrorCode
from luxio.common.error_codes import Error
import pathlib, os
import pandas as pd
import numpy as np
import abc
from typing import List, Dict, Tuple
from functools import reduce

class TraceParser(abc.ABC):
    def __init__(self):
        self.df = None

    @abc.abstractmethod
    def _initialize(self):
        raise Error(ErrorCode.NOT_IMPLEMENTED)

    @abc.abstractmethod
    def _finalize(self):
        raise Error(ErrorCode.NOT_IMPLEMENTED)

    @abc.abstractmethod
    def parse(self) -> None:
        """
        Parse a Trace and return extracted variables
        """
        raise Error(ErrorCode.NOT_IMPLEMENTED)

    @abc.abstractmethod
    def standardize(self) -> None:
        """
        Rename columns, create derivative columns, validate entries
        """
        raise Error(ErrorCode.NOT_IMPLEMENTED)

    def preprocess(self, out=None):
        """
        Parse and standardize data
        """
        self._initialize()
        self.parse()
        self.standardize()
        self.to_csv(out)
        self._finalize()
        return self.df

    def _minimum_features(self, file, name="features.csv") -> List[str]:
        """
        Load the minimum set of features that a trace is required to have. Some traces do not
        always report the same amount of features, which makes standardizing more difficult.

        Returns a list of column names
        """

        script_path = pathlib.Path(file).parent.absolute()
        path = os.path.join(script_path, name)
        features = list(pd.read_csv(path, header=None).iloc[:,0])
        return features

    def _project(self, file, name="mapping.csv"):
        """
        Rename a set of columns using the CSV at map_path
        """
        script_path = pathlib.Path(file).parent.absolute()
        path = os.path.join(script_path, name)
        mappings = pd.read_csv(path)
        map = {orig_name : new_name for orig_name, new_name in zip(list(mappings["original"]), list(mappings["new"]))}
        self.df = self.df.rename(columns=map)

    @staticmethod
    def combine(dfs):
        """
        Intersect the columns of dataframes
        """
        if len(dfs) == 1:
            return dfs[0]
        common = reduce(lambda df1,df2 : np.intersect1d(df1.columns, df2.columns), dfs)
        dfs = [df[common] for df in dfs]
        df = pd.concat(dfs)
        return df

    def to_csv(self, path):
        """
        Save the preprocessed dataframe to a CSV
        """
        if path == None:
            return
        self.df.to_csv(path, index=False)
