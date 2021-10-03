
from luxio.external_clients.json_client import JSONClient
from luxio.common.configuration_manager import *
from typing import List, Dict, Tuple
import pandas as pd

class Mapper:
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
        Takes in an I/O identifier and produces a ranked list of candidate sslos to pass to the
        resource resolver.
        """
        self._initialize()
        self.conf.timer.resume()
        app_classifier = self.conf.app_classifier
        #Get the common set of scores between app classifier and sslos
        scores = list(app_classifier.app_classes[app_classifier.scores].columns.intersection(app_classifier.app_sslo_mapping.columns))
        #Get candidate AppClasses
        fitnesses = app_classifier.get_fitnesses(io_identifier)
        fitnesses = fitnesses[fitnesses.magnitude > self.conf.min_fitness_thresh]
        if len(fitnesses) == 0:
            print("There were no existing classes for this application. Exiting.")
            exit(1)
        #Get candidate sslos from the AppClasses
        name_mapping = {score:f'sslo_{score}' for score in scores if score in app_classifier.app_sslo_mapping.columns}
        reverse_mapping = {f'sslo_{score}':score for score in scores if score in app_classifier.app_sslo_mapping.columns}
        sslo_scores = list(name_mapping.values())
        fit_v_cov = pd.merge(fitnesses[scores + ['app_id']], app_classifier.app_sslo_mapping.rename(columns=name_mapping), on='app_id')
        fitnesses = fit_v_cov[scores]
        ranked_sslos = fit_v_cov[sslo_scores + ['sslo_id']].rename(columns=reverse_mapping)
        #Multiply fitness and coverage and get magnitude
        ranked_sslos.loc[:,scores] = fitnesses[scores].to_numpy() * ranked_sslos[scores].to_numpy()
        ranked_sslos['satisfaction'] = app_classifier.get_magnitude(ranked_sslos)
        #Select sslos above threshold
        ranked_sslos = ranked_sslos.groupby(["sslo_id"]).max().reset_index()
        ranked_sslos[ranked_sslos.satisfaction >= self.conf.min_satisfaction_thresh]
        #Sort the sslos in descending order
        ranked_sslos.sort_values("satisfaction")
        #Verify that at least on sslo was selected
        if len(ranked_sslos) == 0:
            print("There were no existing sslos that were able to satisfy the application. Exiting.")
            exit(1)
        #Return the ranked set of sslos
        self.conf.timer.pause().log("MappingAlgorithm")
        self._finalize()
        return ranked_sslos
