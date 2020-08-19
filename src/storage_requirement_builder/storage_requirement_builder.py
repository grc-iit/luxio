from external_clients.json_client import JSONClient
from utils.mapper_manager import MapperManager


class StorageRequirementBuilder:
    def __init__(self) -> None:
        pass

    def _initialize(self) -> None:
        pass

    def _finalize(self) -> None:
        pass

    def run(self, io_requirement: dict) -> dict:
        self._initialize()
        output = JSONClient().load("storage_requirement.json")
        mapper = MapperManager()
        mapper.run(io_requirement, output)
        self._finalize()
        return output
