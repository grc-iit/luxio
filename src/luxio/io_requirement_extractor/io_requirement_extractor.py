from luxio.mapper.models import AppClassifier
from luxio.external_clients.json_client import JSONClient
from luxio.common.configuration_manager import *
from luxio.common.enumerations import *
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

        #TODO: remove this hack
        #ac = AppClassifier.load(self.conf.app_classifier_path)
        #db.put({"app_classifier", ac})

    def run(self) -> pd.DataFrame:
        """
        Reading the trace input and then mapping it to the corresponding io requirement.
        It will return the io requirement
        :return: dict
        """

        self._initialize()

        #Acquire historical trace data
        self.conf.timer.resume()
        darshan_parser = TraceParserFactory.get_parser(TraceParserType.DARSHAN)
        all_features = darshan_parser.preprocess()
        self.conf.timer.pause().log("Preprocessing")

        #Load I/O behavior classifier model
        self.conf.timer.resume()
        self.conf.app_classifier = AppClassifier.load(self.conf.app_classifier_path)
        self.conf.timer.pause().log("LoadAppClassifier")

        #Feature projection
        self.conf.timer.resume()
        io_identifier = all_features[self.conf.app_classifier.features + self.conf.app_classifier.mandatory_features]
        self.conf.timer.pause().log("FeatureProjection")

        #Return the I/O Identifier
        self._finalize()
        return io_identifier

    def _finalize(self) -> None:
        pass
