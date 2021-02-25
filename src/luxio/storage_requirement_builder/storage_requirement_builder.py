
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
        self.thresh = .25

    def _finalize(self) -> None:
        pass

    def run(self, io_identifier:pd.DataFrame) -> pd.DataFrame:
        """
        Takes in an I/O identifier and produces a ranked list of candidate QoSAs to pass to the
        resource resolver.
        """
        self._initialize()
        #Get the fitness vector of the IOIdentifier to all of the classes
        fitnesses = app_classifier.get_fitnesses(io_identifier)
        #Multiply fitness and coverage
        ranked_qosas = app_classifier.qosas.copy()
        ranked_qosas.loc[:,app_classifier.scores] = fitnesses[app_classifier.scores].to_numpy() * ranked_qosas[app_classifier.scores].to_numpy()
        #Select the best 20 qosas
        ranked_qosas = ranked_qosas.groupby(["qosa_id"]).max().nlargest(20, "magnitude")
        #Sort the QoSAs in descending order
        ranked_qosas.sort_values("magnitude")
        #Return the ranked set of QoSAs
        self._finalize()
        return ranked_qosas

