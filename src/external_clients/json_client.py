
import json
from jsonschema import validate

class JSONClient:
    def __init__(self):
        pass

    def load(self, filename: str) -> dict:
        with open(filename) as fp:
            dict = json.load(fp)
        return dict

    def save(self, object, filename):
        with open(filename) as fp:
            json.dump(object, fp)