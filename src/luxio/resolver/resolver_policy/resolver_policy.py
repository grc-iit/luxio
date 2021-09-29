
from abc import ABC, abstractmethod
from luxio.common.configuration_manager import ConfigurationManager
import pandas as pd

class ResolverPolicy(ABC):
    """
    A class used to Query the scheduler and keep track of existing resources
    """

    @abstractmethod
    def rank(self, deployments:pd.DataFrame) -> pd.DataFrame:
        return
