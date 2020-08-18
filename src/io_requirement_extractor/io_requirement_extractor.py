from src.external_clients.json_client import JSONClient
from src.utils.mapper_manager import MapperManager


class IORequirementExtractor:
    def __init__(self):
        pass

    def _initialize(self):

        pass

    def run(self):

        #set the value from parser into input.
        input = None
        output = JSONClient().load("io_requirement.json")
        mapper = MapperManager()
        mapper.run(input, output)
        return output

    def _finalize(self):
        pass

