from luxio.storage_requirement_builder.models import AppClassifier
from luxio.external_clients.json_client import JSONClient
from luxio.utils.mapper_manager import MapperManager
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

    def run(self) -> pd.DataFrame:
        """
        Reading the trace input and then mapping it to the corresponding io requirement.
        It will return the io requirement
        :return: dict
        """

        self._initialize()

        #Acquire historical trace data
        darshan_parser = TraceParserFactory.get_parser(TraceParserType.DARSHAN)
        all_features = darshan_parser.preprocess()
        
        #TODO: Parse the Job Spec

        #Load I/O behavior classifier model
        self.conf.app_classifier = AppClassifier.load(self.conf.app_classifier_path)

        #Feature projection
        io_identifier = all_features[self.conf.app_classifier.features]

        #Return the I/O Identifier
        self._finalize()
        return io_identifier

    def _finalize(self) -> None:
        pass
