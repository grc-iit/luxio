from abc import ABC, abstractmethod

from src.external_clients.json_client import JSONClient
from src.utils.mapper_manager import MapperManager


class StorageConfigurator(ABC):
    def __init__(self):
        pass

    def _initialize(self):
        pass

    def _finalize(self):
        pass

    def run(self, storage_requirement):
        self._initialize()
        output = self.load_json()
        mapper = MapperManager()
        mapper.run(storage_requirement, output)
        self._finalize()
        return output

    @abstractmethod
    def load_json(self):
        pass