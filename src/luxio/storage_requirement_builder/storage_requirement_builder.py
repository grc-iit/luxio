
from luxio.external_clients.json_client import JSONClient
from luxio.utils.mapper_manager import MapperManager
from luxio.common.configuration_manager import *
from typing import List, Dict, Tuple
import pandas as pd

class StorageRequirementBuilder:
    """
    Build the storage requirements for Luxio
    """
    def __init__(self) -> None:
        pass

    def _initialize(self) -> None:
        self.conf = ConfigurationManager.get_instance()

    def _finalize(self) -> None:
        pass

    def run(self, io_identifier: pd.DataFrame) -> List[Tuple[int, pd.DataFrame]]:
        """
        Takes in an I/O identifier and produces a ranked list of candidate QoSAs to pass to the
        resource resolver.
        """
        self._initialize()
        #qosas = self.conf.app_classifier.rank_qosas(io_identifier)
        self._finalize()
        return [(1, io_identifier)]
