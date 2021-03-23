from luxio.mapper.models import AppClassifier
from luxio.external_clients.json_client import JSONClient
from luxio.common.configuration_manager import *
from luxio.common.enumerations import *
from luxio.database.database import *
from luxio.io_requirement_extractor.trace_parser.trace_parser_factory import *
import pandas as pd
import os

import pprint, warnings
pp = pprint.PrettyPrinter(depth=6)

class IORequirementExtractor:
    """
    Extract the i/o requirement for Luxio
    """
    def __init__(self):
        self.conf = None

    def _initialize(self) -> None:
        self.conf = ConfigurationManager.get_instance()
        self.db = DataBase.get_instance()

    def run(self) -> pd.DataFrame:
        """
        Reading the trace input and then mapping it to the corresponding io requirement.
        It will return the io requirement
        :return: dict
        """

        self._initialize()

        #Acquire historical trace data
        self.conf.timer.resume()
        features = []
        for trace in self.conf.traces:
            parser = TraceParserFactory.get_parser(trace['type'])
            features.append(parser.preprocess(trace))
        io_traits_vec = pd.concat(features).mean().to_frame().transpose()
        self.conf.io_traits_vec = io_traits_vec
        self.conf.timer.pause().log("Preprocessing")

        #Load I/O behavior classifier model
        self.conf.timer.resume()
        self.conf.app_classifier = self.db.get("app_classifier")
        self.conf.storage_classifier = self.db.get("storage_classifier")
        self.conf.timer.pause().log("DownloadModels")

        #Feature projection
        self.conf.timer.resume()
        io_identifier = io_traits_vec[self.conf.app_classifier.features + self.conf.app_classifier.mandatory_features]
        io_identifier = self.conf.app_classifier.standardize(io_identifier)
        self.conf.timer.pause().log("FeatureProjection")

        #Return the I/O Identifier
        self._finalize()
        return io_identifier

    def _finalize(self) -> None:
        pass
