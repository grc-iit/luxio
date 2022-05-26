import argparse
from luxio.common.configuration_manager import *

class ArgumentParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--conf", default=None, help="Luxio configuration file")
        self.parser.add_argument("-m", default='full', help="Mode (options include 'io', 'stor', 'conf', and 'full')")
        self.parser.add_argument("-j", default=None, help="Job spec (json)")
<<<<<<< HEAD:src/luxio/utils/argument_parser.py
        self.parser.add_argument("-t", default="sample/hacc_io_read.darshan", help="Trace file (Darshan)")
        self.parser.add_argument("-i", default="sample/io_req_output.json", help="I/O Requirements file/template")
        self.parser.add_argument("-s", default="sample/stor_req_output.json", help="Storage Requirements file/template")
        self.parser.add_argument("-c", default="sample/stor_req_conf_output.json", help="Storage Configuration template")
=======
        self.parser.add_argument("-t", default=None, help="Trace file (Darshan)")
        self.parser.add_argument("-i", default="resources/io_req_output.json", help="I/O Requirements file/template")
        self.parser.add_argument("-s", default="resources/stor_req_output.json", help="Storage Requirements file/template")
        self.parser.add_argument("-c", default="resources/stor_req_conf_output.json", help="Storage Configuration template")
>>>>>>> 0a8f40db336dc878b365def639aab7c84c63a832:src/utils/argument_parser.py
        self.parser.add_argument("-o", default=None, help="Output file")
        self.parser.add_argument("--app-classifier", default="sample/app_classifier/app_class_model.pkl", help="The Application I/O Behavior Classifier model")
        self.parser.add_argument(
            "--redis-host", default="localhost", help="Host to use for redis"
        )
        self.parser.add_argument("--redis-port", default=6379, help="Port to use for redis")

    def run(self) -> ConfigurationManager:
        args = self.parser.parse_args()
        conf = ConfigurationManager.get_instance()
<<<<<<< HEAD:src/luxio/utils/argument_parser.py
        if args.conf is None:
            conf.job_spec=args.j #"sample/job_info.json"
            conf.darshan_trace_path=args.t #"sample/sample.darshan"
            conf.io_req_out_path=args.i #"sample/io_req_output.json"
            conf.storage_req_out_path=args.s #"sample/stor_req_output.json"
            conf.storage_req_config_out_path=args.c #"sample/stor_req_conf_output.json"
            conf.db_type = KVStoreType.REDIS
            conf.db_addr=args.redis_host
            conf.db_port=str(args.redis_port)
            conf.run_mode=args.m
            conf.output_file=args.o
            conf.app_classifier_path=args.app_classifier
        else:
            conf = conf.load(args.conf)
=======
        conf.job_spec=args.j #"resources/job_info.json"
        conf.darshan_trace_path=args.t #"resources/sample.darshan"
        conf.io_req_out_path=args.i #"resources/io_req_output.json"
        conf.storage_req_out_path=args.s #"resources/stor_req_output.json"
        conf.storage_req_config_out_path=args.c #"resources/stor_req_conf_output.json"
        conf.db_type = KVStoreType.REDIS
        conf.db_addr=args.redis_host
        conf.db_port=str(args.redis_port)
        conf.run_mode=args.m
        conf.output_file=args.o
>>>>>>> 0a8f40db336dc878b365def639aab7c84c63a832:src/utils/argument_parser.py
        self._finalize()
        return conf

    def _finalize(self) -> None:
        pass
