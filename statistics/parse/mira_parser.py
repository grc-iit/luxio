from .generic_trace_parser import GenericTraceParser
import pandas as pd

class MiraParser(GenericTraceParser):
    def __init__(self, path, map_path):
        self.path = path
        self.df = pd.read_csv(path)
        self.map_path = map_path

    def clean(self):
        return

    def standardize(self):
        self._project(self.map_path) 
