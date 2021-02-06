
from luxio.external_clients.json_client import JSONClient
from luxio.utils.mapper_manager import MapperManager
from luxio.common.configuration_manager import *


class StorageRequirementBuilder:
    """
    Build the storage requirements for Luxio
    """
    def __init__(self) -> None:
        pass

    def _initialize(self) -> None:
        pass

    def _finalize(self) -> None:
        pass

    def run(self, io_requirement: dict) -> dict:
        """
        Mapping the given i/o requirements to its corresponding storage requirement.
        And then return the storage requirement.
        :param io_requirement: dict
        :return: dict
        """
        self._initialize()
        conf = ConfigurationManager.get_instance()
        storage_requirement = JSONClient().load(conf.storage_req_out_path)
        mapper = MapperManager()
        mapper.run(io_requirement, storage_requirement)
        self._finalize()
        return storage_requirement
