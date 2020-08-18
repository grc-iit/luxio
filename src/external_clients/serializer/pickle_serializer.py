
import abc
import pickle
from common.error_codes import *
from external_clients.serializer.serializer import *

class PickleSerializer(Serializer):
    def __init__(self):
        pass

    def serialize(self, json_dict: dict) -> str:
        try:
            return pickle.dumps(json_dict)
        except pickle.PicklingError as child_error:
            raise Error(ErrorCode.INVALID_SERIAL_DICT, child_error).format("PickleSerializer")

    def deserialize(self, serial: str) -> dict:
        try:
            return pickle.loads(serial)
        except pickle.UnpicklingError as child_error:
            raise Error(ErrorCode.INVALID_DESERIAL_DICT, child_error).format("PickleSerializer")