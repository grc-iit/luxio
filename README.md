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
* Clever

### Darshan

```{bash}
#git clone https://xgitlab.cels.anl.gov/darshan/darshan   
wget ftp://ftp.mcs.anl.gov/pub/darshan/releases/darshan-3.2.1.tar.gz
tar -xzf darshan-3.2.1.tar.gz
cd darshan-3.2.1/darshan-util
./configure --prefix=</path/to/wherever> --enable-pydarshan --enable-shared  
make install
```

### Redis
```{bash}
wget http://download.redis.io/redis-stable.tar.gz
tar xvzf redis-stable.tar.gz
cd redis-stable
make
make test
cp src/redis-server /path/to/bin
cp src/redis-cli /path/to/bin
```

### Clever

```{bash}
git clone https://github.com/lukemartinlogan/clever.git
cd clever
python3 -m pip install -r requirements.txt  
python3 setup.py develop --user
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
pytest -k unit
```

To run only integration tests:
```{bash}
cd /path/to/luxio
pytest -k integration
```

## License
