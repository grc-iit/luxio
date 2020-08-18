
from common.enumerations import *
from common.error_codes import *
from external_clients.serializer.serializer import *
from external_clients.serializer.pickle_serializer import *
from external_clients.serializer.cereal_serializer import *
from external_clients.serializer.protobuf_serializer import *

class SerializerFactory:

    CEREAL=1
    PICKLE=2
    PROTOBUF=3

    _serial_classes = {
        CEREAL: CerealSerializer,
        PICKLE: PickleSerializer,
        PROTOBUF: ProtobufSerializer
    }

    def __init__(self):
        pass

    def get(self, serial_id: int) -> Serializer:
        try:
            return SerializerFactory._serial_classes[serial_id]()
        except:
            raise Error(ErrorCode.INVALID_SERIAL_ID).format(serial_id)
