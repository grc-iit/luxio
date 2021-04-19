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
        dev_counts = [8,16]
        networks = [10,40]
        workloads = ["compute_intense", "compute_heavy", "balanced", "data_heavy", "data_intense"]

        results = []
        for network in networks:
            for dev_count in dev_counts:
                for workload in workloads:
                    conf_json = {
                      "job_spec_path": f"sample/job_specs/resource_aware/{dev_count}hdd_{network}g.json",
                      "price_conf": "sample/price_confs/avail_price.json",
                      "timer_log_path": "datasets/luxio_timer_log.csv",
                      "check_db": False,
                      "db_type": "REDIS",
                      "db_addr": "127.0.0.1",
                      "db_port": "6379",
                      "traces": [
                        {
                          "path": f"sample/traces/ior/{workload}.json",
                          "type": "DARSHAN_DICT"
                        }
                      ]
                    }
                    conf.update(conf_json)

                    job_spec = LUXIO().run()
                    runtime, util = RuntimeEmulator().run(conf.io_traits_vec, job_spec['qosa'])
                    results.append({"runtime": runtime, "util": util, "dev_count": dev_count, "network": network, "workload": workload})
        df = pd.DataFrame(results)
        df.to_csv("datasets/results/resource_awareness_internal.csv")

if __name__ == "__main__":
    unittest.main()
