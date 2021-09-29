
from .resolver_policy import ResolverPolicy
from luxio.external_clients.json_client import JSONClient
from luxio.common.configuration_manager import ConfigurationManager

import pandas as pd
import numpy as np

class MaxUtilization(ResolverPolicy):
    def rank(self, deployments:pd.DataFrame) -> pd.DataFrame:
        deployments.sort_values(by="price", inplace=True, ascending=True)
        return deployments
