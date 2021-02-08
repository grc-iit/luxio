
class Error(BaseException):
    def __init__(self, error_code, child_error = None):
        self._error_code = error_code["id"]
        if child_error is None:
            self._error_message = "{}".format(error_code["msg"])
        else:
            self._error_message = str(child_error) + "\n{}".format(error_code["msg"])

    def format(self, *error_code_args):
        """
        Formatted the error message
        :param error_code_args:
        :return: Error
        """
        self._error_message = self._error_message.format(error_code_args)
        return self

    def __repr__(self):
        return {'error_code': self._error_code, 'error_message': self._error_message}

    def __str__(self):
        return self._error_message

class ErrorCode:
    """
    A class shows all the error code in Luxio
    """
    NONE = {"id": 0, "msg": "SUCCESSFUL"}

    #General error code
    NOT_IMPLEMENTED = {"id": 1, "msg": "{} is not implemented"}
    TOO_MANY_INSTANCES = {"id": 2, "msg": "{} is a singleton class, but has been initialized more than once"}
    INVALID_ENUM = {"id": 3, "msg": "{}: {} is not a valid enum"}

    #Serializers module error code
    INVALID_SERIAL_DICT = {"id": 1000, "msg": "{}: Could not serialize dict"}
    INVALID_DESERIAL_STR = {"id": 1001, "msg": "{}: Could not deserialize string"}

    #MapperManager error code
    MAPPER_EXEC_ERROR = {"id": 2000, "msg": "MapperManager: Generic execution error in {}: {}"}

    #Trace Parser module error code
    PARSER_ID = {"id": 4000, "msg": "ParserFactory: Invalid Parser ID: {}"}

    # DataBase module error code
    INVALID_KV_STORE_TYPE = {"id": 3000, "msg": "KVStoreFactory: Invalid key-value store type: {}"}
    REDISDB_STORE_ERROR = {"id": 3001, "msg": "RedisDB: Failed to store data into Redis"}
    REDISDB_GET_ERROR = {"id": 3002, "msg": "RedisDB: Failed to get data from Redis"}
    REDISDB_QUERY_ERROR = {"id": 3003, "msg": "RedisDB: Failed to query data from Redis"}
