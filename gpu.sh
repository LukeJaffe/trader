#!/bin/bash

#XSOCK=/tmp/.X11-unix
#XAUTH=/tmp/.docker.xauth-n
#xauth nlist | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -

gpudocker run \
 -i classes:python -c /bin/bash \
 --docker_args="-it --rm \
 --name hw_`date +%F_%H-%M-%S` \
 --net=host \
 -v ${HOME}/.vimrc:/home/username/.vimrc \
 -v ${HOME}/.vim:/home/username/.vim \
 -v `pwd`/work:/home/username/work \
 -e XAUTHORITY=$XAUTH \
 -e DISPLAY=$DISPLAY \
 -e QT_X11_NO_MITSHM=1 \
 -e QT_CRASH_OUTPUT=/home/username/qt_crash.log \
 -w /home/username/work \
 --privileged"
