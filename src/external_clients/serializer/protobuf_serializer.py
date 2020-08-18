
import abc
import pickle
from common.error_codes import *
from external_clients.serializer.serializer import *

class ProtobufSerializer(Serializer):
    def __init__(self):
        pass

    def serialize(self, json_dict: dict) -> str:
        pass

    def deserialize(self, serial: str) -> dict:
        pass