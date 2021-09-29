
import abc
import pickle
from luxio.common.error_codes import *
from luxio.external_clients.serializer.serializer import *

class PickleSerializer(Serializer):
    """
    Using pickle library to serialize and deserialize a python object structure
    """
    def __init__(self):
        pass

    def serialize(self, json_dict: dict) -> str:
        """
        Serialize a python dictionary into a binary serialization of that dictionary
        :param json_dict: dict
        :return: str
        """
        try:
            return pickle.dumps(json_dict)
        except pickle.PicklingError as child_error:
            raise Error(ErrorCode.INVALID_SERIAL_DICT, child_error).format("PickleSerializer")

    def deserialize(self, serial: str) -> dict:
        """
        Deserialize a binary serialization of a dictionary into its original dictionary
        :param serial: str
        :return: dict
        """
        try:
            return pickle.loads(serial)
        except pickle.UnpicklingError as child_error:
            raise Error(ErrorCode.INVALID_DESERIAL_DICT, child_error).format("PickleSerializer")
