from io_requirement_extractor.io_requirement_extractor import IORequirementExtractor
from storage_configurator.storage_configurator import *
from storage_requirement_builder.storage_requirement_builder import *

class LUXIO:
    def __init__(self):
        pass

    def _initialize(self) -> None:
        pass

    def run(self) -> dict:
        self._initialize()
        # run io requirement extractor
        extractor = IORequirementExtractor()
        io_requirement = extractor.run()
        #
        builder = StorageRequirementBuilder()
        storage_requirement = builder.run(io_requirement)
        #
        configurator = StorageConfigurator()
        configuration = configurator.run(storage_requirement)
        self._finalize()
        return configuration

    def _finalize(self) -> None:
        pass


if __name__ == '__main__':
    """
    The main method to start the benchmark runtime.
    """
    tool = LUXIO()
    tool.run()
    exit(0)
