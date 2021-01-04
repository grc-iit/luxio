import argparse
from common.configuration_manager import *

class ArgumentParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("-m", default='full', help="Mode (options include 'io', 'stor', 'conf', and 'full')")
        self.parser.add_argument("-j", default=None, help="Job spec (json)")
        self.parser.add_argument("-t", default=None, help="Trace file (Darshan)")
        self.parser.add_argument("-i", default="sample/io_req_output.json", help="I/O Requirements file/template")
        self.parser.add_argument("-s", default="sample/stor_req_output.json", help="Storage Requirements file/template")
        self.parser.add_argument("-c", default="sample/stor_req_conf_output.json", help="Storage Configuration template")
        self.parser.add_argument("-o", default=None, help="Output file")
        #     self.parser.add_argument(
        #         "-r",
        #         "--enable-redis",
        #         help="""Enable use of
        # Redis""",
        #         action="store_true",
        #     )
        self.parser.add_argument(
            "--redis-host", default="localhost", help="Host to use for redis"
        )
        self.parser.add_argument("--redis-port", default=6379, help="Port to use for redis")
        # self.parser.add_argument(
        #     "--redis-db", default=0, help="Database number to use for redis"
        # )
        # self.parser.add_argument(
        #     "-v",
        #     "--verbose",
        #     help="Toggle extra detail in\noutput",
        #     action="store_true",
        # )

    def run(self) -> ConfigurationManager:
        args = self.parser.parse_args()
        conf = ConfigurationManager.get_instance()
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
        self._finalize()
        return conf

    def _finalize(self) -> None:
        pass
