
import abc

class Serializer(abc.ABC):
    """
    A class used to serialize and deserialize the given python object structure
    """
    @abc.abstractmethod
    def serialize(self, json_dict):
        """
        Serialize a python dictionary into a binary serialization of that dictionary
        :param json_dict: dict
        :return: str
        """
        pass

    @abc.abstractmethod
    def deserialize(self, serial):
        """
        Deserialize a binary serialization of a dictionary into its original dictionary
        :param serial: str
        :return: dict
        """
        pass
