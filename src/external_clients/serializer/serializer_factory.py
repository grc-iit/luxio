
from src.common.enumerations import *
from src.common.error_codes import *
from src.external_clients.serializer.serializer import *
from src.external_clients.serializer.pickle_serializer import *
from src.external_clients.serializer.message_pack_serializer import *

class SerializerFactory:
    _serial_classes = {
        SerializerType.PICKLE: PickleSerializer,
        SerializerType.MSGPACK: MessagePackSerializer
    }

    def __init__(self):
        pass

    @staticmethod
    def get(serial_id: int) -> Serializer:
        try:
            return SerializerFactory._serial_classes[serial_id]()
        except:
            raise Error(ErrorCode.INVALID_SERIAL_ID).format(serial_id)
