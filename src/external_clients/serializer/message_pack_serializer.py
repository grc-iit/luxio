
import abc
import msgpack

from common.error_codes import *
from external_clients.serializer.serializer import *

class MessagePackSerializer(Serializer):
    def __init__(self):
        pass

    def serialize(self, json_dict: dict) -> str:
        try:
            return msgpack.dumps(json_dict)
        except:
            raise Error(ErrorCode.INVALID_SERIAL_DICT).format("MessagePackSerializer")

    def deserialize(self, serial: str) -> dict:
        try:
            return msgpack.loads(serial)
        except:
            raise Error(ErrorCode.INVALID_DESERIAL_STR).format("MessagePackSerializer")