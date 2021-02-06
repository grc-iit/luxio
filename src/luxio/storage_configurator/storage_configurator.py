
from abc import ABC, abstractmethod
from luxio.external_clients.json_client import JSONClient
from luxio.utils.mapper_manager import MapperManager

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

    def run(self, storage_requirement: dict) -> dict:
        """
        Mapping the storage requirement to its corresponding storage configuration.
        And then return the storage configuration.
        :param storage_requirement: dict
        :return: dict
        """
        self._initialize()
        output = self.load_json()
        mapper = MapperManager()
        mapper.run(storage_requirement, output)
        self._finalize()
        return output

    @abstractmethod
    def load_json(self) -> dict:
        """
        Load json file and return a python dictionary
        :return: dict
        """
        pass
