class ConfigurationManager:
    _instance = None

    @staticmethod
    def get_instance():
        """ Static access method. """
        if ConfigurationManager._instance is None:
            ConfigurationManager()
        return ConfigurationManager._instance

    def __init__(self):
        pass

    """
    Configuration Varibales go here.
    """
    def load(self) -> None:
        """
        This method loads up a json of configurations and sets it in the class.
        """
        pass