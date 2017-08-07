FROM ubuntu:16.04
MAINTAINER Luke Jaffe <lukejaffe1@gmail.com>

RUN apt-get update
RUN apt-get install -y vim git python3-dev python3-pip

RUN pip3 install http://download.pytorch.org/whl/cu75/torch-0.1.12.post2-cp35-cp35m-linux_x86_64.whl 
RUN pip3 install torchvision

RUN pip3 install matplotlib

RUN pip3 install quandl

RUN pip3 install ipython
