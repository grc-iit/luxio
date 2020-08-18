
import abc

class Serializer(abc.ABC):


    """
    Serialize a python dictionary and
    return a binary serialization of
    that dictionary.
    """

    @abc.abstractmethod
    def serialize(self, json_dict):
        pass

    """
    Input a binary serialization of a
    dictionary and return the original
    dictionary.
    """

    @abc.abstractmethod
    def deserialize(self, serial):
        pass
