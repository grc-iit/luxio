# Definition
Storage Deployment Schemas explain how to deploy storage so that it can be done simply and automatically using our OrangeFS deployment scripts. Ideally, other services such as Lustre would be supported also. The schema for storage deployments is OrangeFS-specific at the moment, and follows the format specified below.
## Schema
{
    "write_model": {
        "dependencies": [],
        "include": "",
        "guard": "self.input['read_heavy']['val'] > self.input['write_heavy']['val']",
        "expr": "self.output['write_model']['val'] = False",
        "val": {
            "SearchTimeoutSecs": 1,
            "TroveMaxConcurrentIO": 4,
            "TCPBufferSend": 53248,
            "TCPBufferReceive": 212992,
            "TCPBindSpecific": 0,
            "FileStuffing": 1,
            "TroveSyncMeta": 0,
            "TroveSyncData": 0,
            "DBCacheSizeBytes": 65536,
            "DBCacheType": 0,
            "DefaultNumDFiles": 4,
            "TroveMethod": 0,
            "SmallFileSize": 1024,
            "DirectIOThreadNum": 40,
            "DirectIOOpsPerQueue": 20,
            "DirectIOTimeout": 250,
            "TreeThreshold": 1,
            "DistrDirServersInitial": 1,
            "DistrDirServersMax": 1,
            "DistrDirSplitSize": 25,
            "Stripe": 0,
            "Stripe Size": 32768
        }
    },

    "read_model": {
        "dependencies": [],
        "include": "",
        "guard": "self.input['read_heavy']['val'] <= self.input['write_heavy']['val']",
        "expr": "self.output['read_model']['val'] = False",
        "val": {
            "SearchTimeoutSecs": 30,
            "TroveMaxConcurrentIO": 4,
            "TCPBufferSend": 53248,
            "TCPBufferReceive": 212992,
            "TCPBindSpecific": 0,
            "FileStuffing": 1,
            "TroveSyncMeta": 1,
            "TroveSyncData": 0,
            "DBCacheSizeBytes": 65536,
            "DBCacheType": 0,
            "DefaultNumDFiles": 4,
            "TroveMethod": 0,
            "SmallFileSize": 1024,
            "DirectIOThreadNum": 40,
            "DirectIOOpsPerQueue": 20,
            "DirectIOTimeout": 250,
            "TreeThreshold": 1,
            "DistrDirServersInitial": 1,
            "DistrDirServersMax": 1,
            "DistrDirSplitSize": 25,
            "Stripe": 1,
            "Stripe Size": 2097152,
            "num_groups": 1,
            "factor": 2
        }
    }
}
