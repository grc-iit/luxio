
from common.enumerations import *
from common.error_codes import *
from external_clients.serializer.serializer import *
from external_clients.serializer.pickle_serializer import *
from external_clients.serializer.message_pack_serializer import *

class SerializerFactory:
    """
    Factory used for creating Serializer object
    """
    def __init__(self):
        pass

    @staticmethod
    def get(type: SerializerType) -> Serializer:
        """
        Return a Serializer object according to the given type.
        If the type is invalid, it will raise an exception.
        :param type: SerializerType
        :return: Serializer
        """
        if type == SerializerType.PICKLE:
            return PickleSerializer()
        elif type == SerializerType.MSGPACK:
            return MessagePackSerializer()
        else:
            raise Error(ErrorCode.INVALID_ENUM).format("SerializerFactory", type)
