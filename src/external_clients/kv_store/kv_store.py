
class KVStore:
    def __init__(self):
        serializer= None
    def put(self,key,value):
        serialized_value = serializer.serialize(value)
        serialized_key = serializer.serialize(key)
        _put_impl(serialized_key,serialized_value)

    @abstractmethod
    def _put_impl(self);