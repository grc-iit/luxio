from src.external_clients.json_client import JSONClient
from src.utils.mapper_manager import MapperManager


class IORequirementExtractor:
    def __init__(self):
        pass

    def _initialize(self) -> None:

        pass

    def run(self) -> dict:
        self._initialize()

        #set the value from parser into input.
        input = None
        output = JSONClient().load("io_requirement.json")
        mapper = MapperManager()
        mapper.run(input, output)
        self._finalize()
        return output

    def _finalize(self) -> None:
        pass

