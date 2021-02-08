
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
        fitnesses = self.conf.app_classifier.get_fitnesses(io_identifier)

        #Rule out fitness vectors below a threshold
        fitnesses = fitnesses[fitnesses.magnitude > self.thresh]

        #Get the coverage values for each of the QoSAs
        ranked_qosas = []
        for idx,fitness in fitnesses.iterrows():
            #Multiply every coverage by fitness (TODO: factor in mandatory variables that cause this to be 0)
            qosas = fitness["qosas"]
            mult_coverages = fitness[self.conf.app_classifier.scores]*qosas[self.conf.app_classifier.scores]
            mult_coverages.index.name = 'qosa_id'
            qosas.loc[:,self.conf.app_classifier.scores] = mult_coverages
            ranked_qosas.append(qosas)
        #If a QoSA appears multiple times, select the maximum coverage
        ranked_qosas = pd.concat(ranked_qosas).groupby(["qosa_id"]).max()
        print(ranked_qosas)

        #Rule out qosas below a threshold
        ranked_qosas = ranked_qosas[ranked_qosas.magnitude > self.thresh**2]

        #If there are no candidate QoSAs, create a new class and calculate coverage against all QoSAs
        if len(ranked_qosas) == 0:
            coverages = self.conf.storage_classifier.get_coverages(io_identifier)
            ranked_qosas = ranked_qosas[ranked_qosas.magnitude > self.thresh**2]

        #Sort the QoSAs in descending order
        ranked_qosas.sort_values("magnitude")

        #Return the ranked set of QoSAs
        self._finalize()
        return ranked_qosas
