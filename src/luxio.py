from io_requirement_extractor.io_requirement_extractor import IORequirementExtractor
from storage_configurator.storage_configurator import StorageRequirementBuilder, StorageConfigurator


class LUXIO:
    def __init__(self):
        pass

    def _initialize(self) -> None:
        pass

    def run(self) -> dict:
        self._initialize()
        # run io requirement extractor
        extractor = IORequirementExtractor()
        output = extractor.run()
        #
        builder = StorageRequirementBuilder()
        requirement = builder.run(output)
        #
        configurator = StorageConfigurator()
        configuration = configurator.run(requirement)
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
