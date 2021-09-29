from luxio.common.configuration_manager import *
from luxio.io_requirement_extractor.trace_parser.darshan import DarshanTraceParser
import pandas as pd

class MiraTraceParser(DarshanTraceParser):
    def __init__(self):
        return

    def _initialize(self):
        self.conf = ConfigurationManager.get_instance()

    def _finalize(self):
        return

    def parse(self, params):
        self._initialize()
        self.df = pd.read_csv(params["mira_path"])
        self._finalize()

    def standardize(self):
        self._project(__file__, name="mira_to_theta.csv")
