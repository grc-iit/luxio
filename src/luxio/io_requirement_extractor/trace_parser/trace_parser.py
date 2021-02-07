
from luxio.common.error_codes import ErrorCode
from luxio.common.error_codes import Error
import pathlib, os
import pandas as pd
import abc
from typing import List, Dict, Tuple

class TraceParser(abc.ABC):

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

    def parse_standardize(self):
        self.parse()
        return self.standardize()

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
