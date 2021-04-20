#Install ubuntu 20.04
FROM ubuntu:20.04
LABEL maintainer="kbateman@hawk.iit.edu"
LABEL version="0.0"
LABEL description="A docker image for Luxio"

#Disable Prompt During Packages Installation
ARG DEBIAN_FRONTEND=noninteractive

#Update ubuntu
RUN apt update

#Install git and python
apt install git python3.8 python3-pip make wget gcc base libjemalloc-dev libz-dev

#Install setuptools
python3 -m pip install setuptools pytest

#Install darshan
cd ${HOME}
git clone https://xgitlab.cels.anl.gov/darshan/darshan
cd darshan/darshan-util
git checkout 932d69994064af315b736ab55f386cffd6289a15
./configure --enable-pydarshan --enable-shared
make -j8
make install
cd pydarshan
python3 setup.py sdist bdist_wheel
python3 -m pip install dist/*.whl  

#Install redis
cd ${HOME}
wget http://download.redis.io/redis-stable.tar.gz
tar xvzf redis-stable.tar.gz
cd redis-stable
make all -j 8
make install
cp src/redis-server /usr/bin
cp src/redis-cli /usr/bin

#Install clever
cd ${HOME}
git clone https://github.com/lukemartinlogan/clever.git
cd clever
python3 -m pip install -r requirements.txt
python3 setup.py develop --user

#Install luxio
cd ${HOME}
git clone https://github.com/lukemartinlogan/luxio.git
python3 -m pip install -r requirements.txt
python3 setup.py develop --user
