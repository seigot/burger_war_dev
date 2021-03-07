FROM dorowu/ubuntu-desktop-lxde-vnc:bionic
LABEL maintainer="seigot<s.takada.3o3@gmail.com>"

RUN apt-get update -q && \
    apt-get upgrade -yq && \
    apt-get install -yq wget curl git build-essential vim sudo lsb-release locales bash-completion tzdata gosu && \
    rm -rf /var/lib/apt/lists/*
RUN useradd --create-home --home-dir /home/ubuntu --shell /bin/bash --user-group --groups adm,sudo ubuntu && \
    echo ubuntu:ubuntu | chpasswd && \
    echo "ubuntu ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

RUN mkdir -p /tmp/ros_setup_scripts_ubuntu
ADD ros-melodic-desktop.sh /tmp/ros_setup_scripts_ubuntu/ros-melodic-desktop.sh
RUN gosu ubuntu /tmp/ros_setup_scripts_ubuntu/ros-melodic-desktop.sh && \
    rm -rf /var/lib/apt/lists/*
ENV USER ubuntu

# update
RUN apt-get update ;\
    apt-get -y upgrade ;

# environment setting
#RUN locale-gen en_US.UTF-8  
#ENV LANG en_US.UTF-8  
#ENV LANGUAGE en_US:en  
#ENV LC_ALL en_US.UTF-8
RUN locale-gen ja_JP.UTF-8
ENV LANG ja_JP.UTF-8
ENV LANGUAGE ja_JP:jp
ENV LC_ALL ja_JP.UTF-8

# tools
RUN apt-get update ;\
    apt-get install -y \
    sudo \
    lsb-release \
    terminator \
    gnome-terminal \
    emacs25 \
    git;

# turtlebot3
RUN apt-get update ;\
    apt-get install -y \
    python-rosdep python-rosinstall python-rosinstall-generator python-wstool build-essential \
    python-pip \
    ros-melodic-turtlebot3 ros-melodic-turtlebot3-msgs ros-melodic-turtlebot3-simulations;

RUN apt-get update ;\
    apt-get install -y python-pip; \
    pip install requests flask; \
    apt install -y \
    ros-melodic-turtlebot3 \
    ros-melodic-turtlebot3-msgs \
    ros-melodic-turtlebot3-simulations \
    ros-melodic-teleop-twist-keyboard \
    ros-melodic-amcl \
    ros-melodic-map-server \
    ros-melodic-move-base\
    ros-melodic-urdf \
    ros-melodic-xacro \
    ros-melodic-compressed-image-transport \
    ros-melodic-rqt-image-view \
    ros-melodic-gmapping \
    ros-melodic-navigation \
    ros-melodic-interactive-markers \
    ros-melodic-hector-gazebo-plugins; \
    apt install -y ros-melodic-aruco-ros; \
    apt install -y python-pil python-pil.imagetk;


# For enemy
RUN apt-get update ;\
    apt install -y \
    libsuitesparse-dev \
    ros-melodic-costmap-converter \
    ros-melodic-libg2o \
    ros-melodic-mbf-costmap-core \
    ros-melodic-mbf-msgs

# log upload
RUN apt-get update ;\
    apt-get install -y \
    python-pandas \
    wmctrl \
    xdotool

# ai lib
#ADD  install_ai_lib.sh  /
#RUN  chmod +x /install_ai_lib.sh
#RUN  /install_ai_lib.sh

# command
ADD  start.sh  /
RUN  chmod +x /start.sh
RUN  /start.sh

