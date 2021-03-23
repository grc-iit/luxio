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
        tool = LUXIO()
        job_spec = tool.run()
        print(job_spec)
        #print(RuntimeEmulator().run(conf.io_traits_vec, job_spec['qosa']))

    def test_luxio_resource_awareness(self):
        conf = ConfigurationManager.get_instance()
        conf_paths = [
            "sample/luxio_confs/resource_utilization/compute_intense.json",
            "sample/luxio_confs/resource_utilization/compute_heavy.json",
            "sample/luxio_confs/resource_utilization/balanced.json",
            "sample/luxio_confs/resource_utilization/data_intense.json",
            "sample/luxio_confs/resource_utilization/data_heavy.json"
        ]
        conf.load(conf_paths[0])
        
        print("HERE!!!")
        job_spec = LUXIO().run()
        print(job_spec)
        print(RuntimeEmulator().run(conf.io_traits_vec, job_spec['qosa']))


if __name__ == "__main__":
    unittest.main()
