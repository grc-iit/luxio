import unittest, pytest
from luxio.common.error_codes import *
from luxio.common.enumerations import *
from luxio.common.configuration_manager import *
from luxio.emulator.runtime_emulator import *
from luxio.external_clients.json_client import *
from luxio.luxio import LUXIO

class TestLuxio(unittest.TestCase):

    def test_luxio_sample(self):
        conf = ConfigurationManager.get_instance()
        conf.load("sample/luxio_confs/basic_conf.json")

        #Schedule the Job
        tool = LUXIO()
        config = tool.run()
        print(config)

        #Emulate the runtime of the Job



if __name__ == "__main__":
    unittest.main()
