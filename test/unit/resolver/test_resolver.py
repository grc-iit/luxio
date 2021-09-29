import unittest, pytest
from luxio.common.error_codes import *
from luxio.common.enumerations import *
from luxio.common.configuration_manager import *
from luxio.mapper.models.app_io_classifier import *
from luxio.resolver.resolver import *
import pandas as pd
import numpy as np

from luxio.common.timer import Timer

class TestResolver(unittest.TestCase):
    def test_resolver_perf(self):
        return
        conf = ConfigurationManager.get_instance()
        conf.load("sample/luxio_confs/basic_conf.json")
        conf.job_spec = {}
        ranked_qosas = pd.read_csv("datasets/stress_tests/qosa.csv")
        ranked_qosas['qosa_id'] = ranked_qosas['deployment_id']
        ranked_qosas['satisfaction'] = 1

        for n_qosas in [10, 20, 30]:
            for n_malleable in [0, 10, 20, 50, 100, 1000, 5000]:
                existing_deployments = ranked_qosas.sample(n_malleable, replace=True)
                existing_deployments['malleable'] = 1
                existing_deployments['status'] = DeploymentStatus.RUNNING
                db = DataBase.get_instance()
                db.put('existing_deployments', existing_deployments)

                t = Timer()
                t.resume()
                resolver = Resolver()
                resolver.run(None, ranked_qosas)
                t.pause()
                print(f"{n_qosas}, {n_malleable}: {t.msec()}")
                t.reset()

if __name__ == "__main__":
    unittest.main()
