# LUXIO

A tool for autconfiguring and provisioning storage.

## Dependencies

Luxio depends on the following:
* SCSPKG (a tool for managing modulefiles)
* Darshan
* Redis

## Installation

### Docker Container
We have a dockerfile to make deployment easier
```
git clone https://github.com/scs-lab/luxio.git
cd luxio
sudo docker build -t luxio-img .
sudo docker run -it --name luxio luxio-img
```

### For Regular Users
```{bash}
git clone https://github.com/scs-lab/luxio.git
cd luxio
bash dependencies.sh
python3 -m pip . --user
module load luxio
```

### For Developers

```{bash}
git clone git@github.com:scs-lab/luxio.git
cd luxio
python3 -m pip install -e .
```

### Uninstallation

```{bash}
python3 -m pip uninstall luxio
```

## Usage

Start the luxio server to maintain resource graph:
```{bash}
luxio-server --conf [/path/to/luxio-conf.json]
```

Run the Luxio scheduler assistance command line tool:
```{bash}
luxio-sched --conf [/path/to/luxio-conf.json]
```

A sample conf is in sample/luxio_confs

### Statistics

<<<<<<< HEAD
```
gdown
```

## Test

In order to run tests, you must have installed Luxio in either developer mode or regular mode.

Before running tests:
```{bash}
#Start Redis
redis-server
#Upload mapper models
luxio-stats -t upload
#Start luxio server
luxio-server --conf [/path/to/luxio-conf.json]
```

To run all tests:
```{bash}
cd /path/to/luxio
pytest
```

To run only unit tests:
```{bash}
cd /path/to/luxio
pytest -k unit -s
```

To run only integration tests:
```{bash}
cd /path/to/luxio
pytest -k integration -s
```
=======
Get the I/O configuration
> storage_config = LuxioBin(conf).run()

## Test

cd /path/to/luxio  
export PYTHONPATH="$(pwd)/src"  
> python3 -m unittest discover -s test/integration/luxio  

> ./luxio.py -m io -o output_file.json -j job_info.json -t resources.darshan  
> ./scripts/print-luxio.sh output_file.json  
The print-luxio.sh script requires the program jq (json query) to be installed, it performs a `jq 'map_values(.val)'` on the given argument

Alternatively, there are now convenience scripts to accomplish the same results as above. Try running:
> cd scripts
> source init-luxio.sh

The scripts in the scripts/ directory include luxio.sh, io-req-extractor.sh, storage-requirement-builder.sh, and storage-configurator.sh, and implement simple wrappers over modes of the Luxio system from luxio.py.

### Script Run Sample Interaction
> cd scripts
> source init-luxio.sh
> ./luxio.sh ../resources/job_info.json ../sample/sample.darshan
> ./luxio.sh ../resources/job_info.json ../sample/sample.darshan -o output-0.json
> ./io-req-extractor.sh ../resources/job_info.json ../sample/sample.darshan
> ./io-req-extractor.sh ../resources/job_info.json ../sample/sample.darshan -o output-1.json
> ./storage-requirement-builder.sh output-1.json
> ./storage-requirement-builder.sh output-1.json output-2.json
> ./storage-configurator.sh ../resources/job_info.json output-1.json output-2.json
> ./storage-configurator.sh ../resources/job_info.json output-1.json output-2.json output-3.json
> diff output-0.json output-3.json
>>>>>>> 0a8f40db336dc878b365def639aab7c84c63a832

## License
