
from luxio.external_clients.json_client import JSONClient
from luxio.common.configuration_manager import ConfigurationManager
from luxio.common.enumerations import *

import pandas as pd
import numpy as np

class InterferenceFactor:
    def interference(self, colocated:pd.DataFrame):
        return 1

    def score(self, deployments:pd.DataFrame) -> pd.DataFrame:
        deployments.loc[deployments.status == DeploymentStatus.RUNNING,'interference'] = 0
        colocated = deployments.status != DeploymentStatus.RUNNING
        deployments.loc[colocated,'interference'] = self.interference(colocated)
        return deployments
