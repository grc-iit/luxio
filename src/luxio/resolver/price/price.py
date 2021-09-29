
from abc import ABC, abstractmethod
from luxio.scheduler.client import LuxioSchedulerClient
from luxio.common.configuration_manager import ConfigurationManager
import pandas as pd

class Price(ABC):
    """
    A class used to Query the scheduler and keep track of existing resources
    """

    @abstractmethod
    def price(self, deployments:pd.DataFrame, scheduler:LuxioSchedulerClient) -> pd.DataFrame:
        return
