from luxio.common.configuration_manager import *
from luxio.io_requirement_extractor.trace_parser.darshan import DarshanTraceParser
import pandas as pd

class DarshanDictParser(DarshanTraceParser):
    def __init__(self):
        return

    def _initialize(self):
        return

    def _finalize(self):
        return

    def parse(self, params):
        path = params["path"]
        self.df = pd.read_json(path)
        return self.df
