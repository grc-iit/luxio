
import abc
import pickle
from src.common.error_codes import *
from src.external_clients.serializer.serializer import *

class PickleSerializer(Serializer):
    def __init__(self):
        pass

    def serialize(self, json_dict: dict) -> str:
        try:
            return pickle.dumps(json_dict)
        except pickle.PicklingError as child_error:
            raise Error(ErrorCode.INVALID_PICKLE_DICT, child_error)

    def deserialize(self, serial: str) -> dict:
        try:
            return pickle.loads(serial)
        except pickle.UnpicklingError as child_error:
            raise Error(ErrorCode.INVALID_UNPICKLE_STR, child_error)