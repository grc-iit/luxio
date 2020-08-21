
import abc
import msgpack

from common.error_codes import *
from external_clients.serializer.serializer import *

class MessagePackSerializer(Serializer):
    """
    Using msgpack library to serialize and deserialize a python object structure
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
            return msgpack.dumps(json_dict)
        except:
            raise Error(ErrorCode.INVALID_SERIAL_DICT).format("MessagePackSerializer")

    def deserialize(self, serial: str) -> dict:
        """
        Deserialize a binary serialization of a dictionary into its original dictionary
        :param serial: str
        :return: dict
        """
        try:
            return msgpack.loads(serial)
        except:
            raise Error(ErrorCode.INVALID_DESERIAL_STR).format("MessagePackSerializer")
