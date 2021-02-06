# LUXIO

A tool for submitting I/O requirements to Flux scheduler.

## Installation

### For Regular Users
```{bash}
cd /path/to/luxio  
python3 -m pip install -r requirements.txt  
python3 setup.py sdist bdist_wheel  
python3 -m pip install dist/*.whl  
rm -r dist build *.egg_info MANIFEST  
```

### For Developers

```{bash}
cd /path/to/luxio  
python3 -m pip install -r requirements.txt  
python3 setup.py develop --user
```

### Uninstallation

```{bash}
python3 -m pip uninstall luxio
```

## Dependencies

Luxio depends on the following:
* Darshan
* Redis

### Darshan

```{bash}
#git clone https://xgitlab.cels.anl.gov/darshan/darshan   
wget ftp://ftp.mcs.anl.gov/pub/darshan/releases/darshan-3.2.1.tar.gz
tar -xzf darshan-3.2.1.tar.gz
cd darshan-3.2.1/darshan-util
./configure --prefix=</path/to/wherever> --enable-pydarshan --enable-shared  
make install
```

## Usage

Run the Luxio scheduler assistance command line tool:
```{bash}
luxio-sched -[params]
```

Import the module in Python:
```{bash}
from luxio import *
```

Configuring Luxio:
```{bash}
conf = ConfigurationManager.get_instance()  
conf.job_spec = "/path/to/job_spec.json"  
conf.darshan_trace_path = "/path/to/darshan_trace.darshan"  
conf.io_req_out_path = "/path/to/io_req_out.json"  
conf.storage_req_out_path = "/path/to/storage_req_out.json"  
conf.storage_req_config_out_path = "/path/to/storage_req_config_out.json"
conf.db_type = KVStoreType.REDIS  
conf.db_addr = "127.0.0.1"  
conf.db_port = "6379"  
```

Get the I/O configuration
```{bash}
storage_config = LUXIO().run()
```

## Test

In order to run tests, you must have installed Luxio in either developer mode or regular mode.  

To run all tests:
```{bash}
cd /path/to/luxio
pytest
```

To run only unit tests:
```{bash}
cd /path/to/luxio
pytest test/integration
```

To run only integration tests:
```{bash}
cd /path/to/luxio
pytest test/integration
```

A sample use of luxio.py
```{bash}
./luxio.py -m io -o output_file.json -j job_info.json -t sample.darshan  
./scripts/print-luxio.sh output_file.json
#The print-luxio.sh script requires the program jq (json query) to be installed, it performs a `jq 'map_values(.val)'` on the given argument
```

Alternatively, there are now convenience scripts to accomplish the same results as above. Try running:
```{bash}
cd scripts
source init-luxio.sh
```

The scripts in the scripts/ directory include luxio.sh, io-req-extractor.sh, storage-requirement-builder.sh, and storage-configurator.sh, and implement simple wrappers over modes of the Luxio system from luxio.py.

### Script Run Sample Interaction
```{bash}
cd scripts
source init-luxio.sh
./luxio.sh ../sample/job_info.json ../sample/sample.darshan
./luxio.sh ../sample/job_info.json ../sample/sample.darshan -o output-0.json
./io-req-extractor.sh ../sample/job_info.json ../sample/sample.darshan
./io-req-extractor.sh ../sample/job_info.json ../sample/sample.darshan -o output-1.json
./storage-requirement-builder.sh output-1.json
./storage-requirement-builder.sh output-1.json output-2.json
./storage-configurator.sh ../sample/job_info.json output-1.json output-2.json
./storage-configurator.sh ../sample/job_info.json output-1.json output-2.json output-3.json
diff output-0.json output-3.json
```

## License
