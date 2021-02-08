
from abc import ABC, abstractmethod
from luxio.external_clients.json_client import JSONClient
from luxio.utils.mapper_manager import MapperManager
import pandas as pd

class StorageConfigurator(ABC):
    """
    A class used to generate the storage configuration according to the given storage requirement
    """
    def __init__(self):
        pass

    def _initialize(self) -> None:
        pass

    def _finalize(self) -> None:
        pass

    def run(self, storage_requirement: pd.DataFrame) -> dict:
        """
        Mapping the storage requirement to its corresponding storage configuration.
        And then return the storage configuration.
        :param storage_requirement: dict
        :return: dict
        """
        self._initialize()
        #Acquire the set of available resources from the scheduler
        #Determine whether or not the qosas in the storage requirement can be satisfied, and if so, how much it costs
        #Choose the lowest-price QoSA that can be satisfied
        self._finalize()
        return None

    @abstractmethod
    def load_json(self) -> dict:
        """
        Load json file and return a python dictionary
        :return: dict
        """
        pass
