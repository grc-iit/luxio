
class Error(BaseException):
    def __init__(self, error_code, child_error = None):
        self._error_code = error_code["id"]
        if child_error is None:
            self._error_message = "{}".format(error_code["msg"])
        else:
            self._error_message = str(child_error) + "\n{}".format(error_code["msg"])

    def format(self, *error_code_args):
        self._error_message = self._error_message.format(error_code_args)
        return self

    def __repr__(self):
        return {'error_code': self._error_code, 'error_message': self._error_message}

    def __str__(self):
        return self._error_message

class ErrorCode:
    NONE = {"id": 0, "msg": "SUCCESSFUL"}

    #General
    NOT_IMPLEMENTED = {"id": 1, "msg": "{} is not implemented"}
    TOO_MANY_INSTANCES = {"id": 2, "msg": "{} is a singleton class, but has been initialized more than once"}

    #Serializers
    INVALID_SERIAL_ID = {"id": 1000, "msg": "SerializerFactory: Invalid serializer ID: {}"}

    INVALID_PICKLE_DICT = {"id": 1001, "msg": "PickleSerializer: Could not serialize dict"}
    INVALID_UNPICKLE_STR = {"id": 1002, "msg": "PickleSerializer: Could not deserialize string"}

    PARSER_ID = {"id": 2000, "msg": "ParserFactory: Invalid Parser ID: {}"}

    # DataBase
    INVALID_KV_STORE_TYPE = {"id": 3000, "msg": "KVStoreFactory: Invalid key-value store type: {}"}
    REDISDB_STORE_ERROR = {"id": 3001, "msg": "RedisDB: Failed to store data into Redis"}
    REDISDB_GET_ERROR = {"id": 3002, "msg": "RedisDB: Failed to get data from Redis"}
    REDISDB_QUERY_ERROR = {"id": 3003, "msg": "RedisDB: Failed to query data from Redis"}
