
import abc
from common.error_codes import *
from external_clients.serializer.serializer import *

class CerealSerializer(Serializer):

    def __init__(self):
        raise Error(ErrorCode.NOT_IMPLEMENTED).format("CerealSerializer")

    def serialize(self, json_dict: dict) -> str:
        raise Error(ErrorCode.NOT_IMPLEMENTED).format("CerealSerializer.serialize()")

    def deserialize(self, serial: str) -> dict:
        raise Error(ErrorCode.NOT_IMPLEMENTED).format("CerealSerializer.deserialize()")
