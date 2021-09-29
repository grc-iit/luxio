from luxio.common.configuration_manager import *
from luxio.io_requirement_extractor.trace_parser.darshan import DarshanTraceParser
from luxio.external_clients.json_client import JSONClient
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
        df_dict = JSONClient().load(path)
        self.df = pd.DataFrame([df_dict], index=[0])
        return self.df

    def standardize(self):
        return
