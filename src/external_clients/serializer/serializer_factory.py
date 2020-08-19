
from common.enumerations import *
from common.error_codes import *
from external_clients.serializer.serializer import *
from external_clients.serializer.pickle_serializer import *
from external_clients.serializer.message_pack_serializer import *

class SerializerFactory:
    def __init__(self):
        pass

    @staticmethod
    def get(serial_id: SerializerType) -> Serializer:
        if serial_id == SerializerType.PICKLE:
            return PickleSerializer()
        elif serial_id == SerializerType.MSGPACK:
            return MessagePackSerializer()
        else:
            raise Error(ErrorCode.INVALID_SERIAL_ID).format(serial_id)
