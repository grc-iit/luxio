
#yas, cereal, boost, thrift-binary, yas-compact, protobuf, thrift-compact, msgpack
#io-req-extractor

from common.enumerations import *
from common.error_codes import *
from external_clients.serializer.serializer import *
from external_clients.serializer.pickle_serializer import *
from external_clients.serializer.message_pack_serializer import *

class SerializerFactory:

    PICKLE=1
    MSGPACK=2

    _serial_classes = {
        PICKLE: PickleSerializer,
        MSGPACK: MessagePackSerializer
    }

    def __init__(self):
        pass

    def get(self, serial_id: int) -> Serializer:
        try:
            return SerializerFactory._serial_classes[serial_id]()
        except:
            raise Error(ErrorCode.INVALID_SERIAL_ID).format(serial_id)
