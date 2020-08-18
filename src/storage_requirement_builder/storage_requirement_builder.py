from src.external_clients.json_client import JSONClient
from src.utils.mapper_manager import MapperManager


class StorageRequirementBuilder:
    def __init__(self):
        pass

    def _initialize(self):
        pass

    def _finalize(self):
        pass

    def run(self, io_requirement):
        self._initialize()
        output = JSONClient().load("storage_requirement.json")
        mapper = MapperManager()
        mapper.run(io_requirement, output)
        self._finalize()
        return output