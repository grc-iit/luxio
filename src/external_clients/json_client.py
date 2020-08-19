
import pprint
import json
from jsonschema import validate

class JSONClient:
    def __init__(self):
        pass

    def load(self, filename: str) -> dict:
        with open(filename) as fp:
            dict = json.load(fp)
        return dict

    def save(self, object, filename) -> None:
        with open(filename) as fp:
            json.dump(object, fp)

    def strip(self, json_dict: dict) -> dict:
        stripped_dict = {}
        for key, val in json_dict.items():
            stripped_dict[key] = val["val"]
        return stripped_dict

    def dumps(self, json_dict: dict) -> None:
        pprint.PrettyPrinter().pprint(json_dict)