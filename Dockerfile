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
RUN apt install git python3.8 python3-pip make wget sudo curl gfortran

#Install setuptools
RUN python3 -m pip install setuptools pytest

#Install luxio
#cd ${HOME}
#git clone https://github.com/lukemartinlogan/luxio.git
#python3 -m pip install .
