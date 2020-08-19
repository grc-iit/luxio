
from common.enumerations import *
from common.error_codes import *
from external_clients.serializer.serializer import *
from external_clients.serializer.pickle_serializer import *
from external_clients.serializer.message_pack_serializer import *

class SerializerFactory:
    def __init__(self):
        pass

    @staticmethod
    def get(type: SerializerType) -> Serializer:
        if type == SerializerType.PICKLE:
            return PickleSerializer()
        elif type == SerializerType.MSGPACK:
            return MessagePackSerializer()
        else:
            raise Error(ErrorCode.INVALID_ENUM).format("SerializerFactory", type)
