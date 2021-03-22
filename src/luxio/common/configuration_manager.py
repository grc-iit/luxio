
from luxio.common.enumerations import *
from luxio.common.error_codes import *
from luxio.common.timer import *
import json
import numpy as np

class ConfigurationManager:
    """
    A singleton class used to provide configuration variables for luxio
    """
    _instance = None

    @staticmethod
    def get_instance():
        """ Static access method. """
        if ConfigurationManager._instance is None:
            ConfigurationManager()
        return ConfigurationManager._instance

    def __init__(self):
        """
        Initialize the configuration variables
        """
        if ConfigurationManager._instance is not None:
            raise Error(ErrorCode.TOO_MANY_INSTANCES).format("ConfigurationManager")
        else:
            ConfigurationManager._instance = self

        self.job_spec = None
        self.job_spec_path = None
        self.darshan_trace_path = None
        self.app_classifier_path = None
        self.storage_classifier_path = None

        self.min_coverage_thresh = 0
        self.min_fitness_thresh = .1
        self.min_satisfaction_thresh = self.min_coverage_thresh * self.min_fitness_thresh
        self.baseline_qosa_id = 0
        self.scheduler_type = SchedulerType.DUMMY
        self.price_type = PriceType.AVAILABILITY
        self.price_conf = None
        self.resolver_policy = ResolverPolicyType.MAX_UTILIZATION
        self.min_performance_improvement = .1 #At least 10% faster than baseline qosa
        self.max_tolerable_interference = np.inf #Tolerate up to X% interference

        self.io_req_out_path = None
        self.storage_req_out_path = None
        self.storage_req_config_out_path = None
        self.check_db = True
        self.db_type = KVStoreType.REDIS
        self.db_addr = "localhost"
        self.db_port = "6379"
        self.db = None

        self.luxio_server_mode = LuxioServerMode.EVENT_BASED
        #self.luxio_server_port = 5555
        #self.luxio_server_addr = "0.0.0.0"
        self.luxio_server_frequency = 20
        self.resource_graph_path = "sample/resource_graphs/resource_graph.json"

        self.run_mode = "full"
        self.output_file = None
        self.serializer_type = SerializerType.PICKLE
        self.timer_log_path = "datasets/luxio_timer_log.csv"
        self.timer = Timer()

    @staticmethod
    def load(filename):
        """
        Read configuration info from the given json file.
        :param filename: str
        :return: ConfigurationManager
        """
        with open(filename) as fp:
            conf_json = json.load(fp)
        conf = ConfigurationManager.get_instance()
        conf.__dict__.update(conf_json)
        if "db_type" in conf_json: conf.db_type = KVStoreType[conf_json['db_type']]
        if "scheduler_type" in conf_json: conf.scheduler_type = SchedulerType[conf_json['scheduler_type']]
        if "price_type" in conf_json: conf.price_type = PriceType[conf_json['price_type']]
        if "resolver_policy" in conf_json: conf.resolver_policy = ResolverPolicyType[conf_json['resolver_policy']]
        if "luxio_server_mode" in conf_json: conf.luxio_server_mode = LuxioServerMode[conf_json['luxio_server_mode']]
        if "serializer_type" in conf_json: conf.serializer_type = SerializerType[conf_json['serializer_type']]
        return conf
