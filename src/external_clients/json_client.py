
import pprint
import json
from jsonschema import validate

class JSONClient:
    def __init__(self):
        pass

    def load(self, filename: str) -> dict:
        """
        Read data from the given json file
        :param filename: str
        :return:dict
        """
        with open(filename) as fp:
            dict = json.load(fp)
        return dict

    def save(self, object, filename) -> None:
        """
        Save the gaven data into a json file
        :param object: dict
        :param filename: str
        """
        with open(filename, "w") as fp:
            json.dump(object, fp)

    def strip(self, json_dict: dict) -> dict:
        """
        Strip all other elements from the given dictionary and
        return a dictionary which only show the value of "val"
        :param json_dict: dict
        :return: dict
        """
        stripped_dict = {}
        for key, val in json_dict.items():
            stripped_dict[key] = val["val"]
        return stripped_dict

    def dumps(self, json_dict: dict) -> None:
        """
        Print a python dictionary
        :param json_dict: dict
        """
        pprint.PrettyPrinter().pprint(json_dict)
