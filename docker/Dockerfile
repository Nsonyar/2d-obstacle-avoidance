#Dockerfile in order to build keras / tf, ROS, Tiago environment
FROM tensorflow/tensorflow:latest-gpu-jupyter
RUN apt-get update

#Ubuntu install of ROS Melodic http://wiki.ros.org/melodic/Installation/Ubuntu
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

#Is necessary to avoid process getting interrupted with asking for timezone
ADD timezone.sh /timezone.sh
RUN /timezone.sh

RUN apt-get update && apt install -y \
    ros-melodic-desktop-full
RUN echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc
RUN source ~/.bashrc

RUN apt-get update && apt install -y \
    python-rosdep \
    python-rosinstall \
    python-rosinstall-generator \
    python-wstool \
    build-essential \
    python-catkin-tools \
    python-rosdep \
    wget

RUN rosdep init
RUN rosdep update

#Installing TIAGo Simulation http://wiki.ros.org/Robots/TIAGo/Tutorials/Installation/TiagoSimulation
RUN mkdir /root/tiago_public_ws

WORKDIR /root/tiago_public_ws

RUN wget https://raw.githubusercontent.com/pal-robotics/tiago_tutorials/kinetic-devel/tiago_public-melodic.rosinstall

RUN rosinstall src /opt/ros/melodic tiago_public-melodic.rosinstall

RUN rosdep install -y --from-paths src --ignore-src --rosdistro melodic --skip-keys="opencv2 opencv2-nonfree pal_laser_filters speed_limit_node sensor_to_cloud hokuyo_node libdw-dev python-graphitesend-pip python-statsd pal_filters pal_vo_server pal_usb_utils pal_pcl pal_pcl_points_throttle_and_filter pal_karto pal_local_joint_control camera_calibration_files pal_startup_msgs pal-orbbec-openni2 dummy_actuators_manager pal_local_planner gravity_compensation_controller current_limit_controller dynamic_footprint dynamixel_cpp tf_lookup opencv3"

RUN source /opt/ros/melodic/setup.bash

WORKDIR /root/tiago_public_ws

RUN catkin config --extend /opt/ros/melodic
RUN catkin build
RUN source devel/setup.bash

#Upgrade pip
RUN pip3 install --upgrade pip

RUN pip install -U rospkg
RUN apt update

#Install python3 dependecies
RUN apt-get update && apt install -y \
    python3-catkin-pkg-modules \
    python3-rospkg-modules \
    python3-empy

WORKDIR /root/tiago_public_ws
RUN catkin build -DCATKIN_ENABLE_TESTING=0
RUN source devel/setup.bash
RUN wstool init
RUN wstool set -y src/geometry2 --git https://github.com/ros/geometry2 -v 0.6.5
RUN wstool up
WORKDIR /root/tiago_public_ws

#Make workspace python3 compatible
WORKDIR /root/tiago_public_ws/src/tiago_tutorials/tiago_pcl_tutorial/cfg
RUN sed -i '11,12 s/010/10/' segment.cfg

#Build python3 workspace
#https://answers.ros.org/question/326226/importerror-dynamic-module-does-not-define-module-export-function-pyinit__tf2/
WORKDIR /root/tiago_public_ws
RUN catkin clean -y
RUN catkin build -DCATKIN_ENABLE_TESTING=0 \
            --cmake-args \
            -DCMAKE_BUILD_TYPE=Release \
            -DPYTHON_EXECUTABLE=/usr/bin/python3 \
            -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m \
            -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so
RUN source devel/setup.bash

#Install gnupg for Data Acquisition 
RUN pip install gnupg

#Install Cryptodome for Data Acquisition
RUN apt-get install python-pip python-crypto build-essential -y
RUN apt-get install python-all-dev python-setuptools python-wheel -y
RUN apt-get install python-crypto-dbg python-crypto-doc python-pip-whl -y
RUN pip install pycryptodomex
RUN ln -s /usr/lib/python2.7/dist-packages/Crypto /usr/lib/python2.7/dist-packages/Cryptodome

#Install opencv-python for Data Acquistion
RUN pip3 install opencv-python

#Install pip for python2
RUN apt-get update && apt install -y \
    python-pip

#Install tqdm for Feature Extraction (python2)
RUN python2 -m pip install tqdm
RUN python2 -c 'import tqdm'

#Install h5py for Feature Extraction  (python2)
RUN python2 -m pip install h5py
RUN python2 -c 'import h5py'

#Install pandas for Feature Extraction  (python2)
RUN python2 -m pip install pandas
RUN python2 -c 'import pandas'

#Install pandas, keras for training 
RUN pip3 install pandas
RUN pip3 install keras

#Install seaborn, tqdm, sklearn for testing
RUN pip3 install seaborn
RUN pip3 install tqdm
RUN pip3 install sklearn

#Install pytest for Testing
#RUN python2 -m pip install pytest
#RUN python2 -c 'import pytest'
#RUN python3 -m pip install pytest
#RUN python3 -c 'import pytest'
RUN pip install -U pytest

#Install setuptools
RUN pip install --upgrade setuptools

#Make catkin workspace for project
#https://answers.ros.org/question/326226/importerror-dynamic-module-does-not-define-module-export-function-pyinit__tf2/
RUN mkdir -p /root/catkin_ws/src
WORKDIR /root/catkin_ws
RUN catkin config --extend /opt/ros/melodic
RUN catkin build
RUN source devel/setup.bash

RUN wstool init
RUN wstool set -y src/geometry2 --git https://github.com/ros/geometry2 -v 0.6.5
RUN wstool up

WORKDIR /root/catkin_ws
RUN catkin clean -y
RUN catkin build -DCATKIN_ENABLE_TESTING=0 \
            --cmake-args \
            -DCMAKE_BUILD_TYPE=Release \
            -DPYTHON_EXECUTABLE=/usr/bin/python3 \
            -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m \
            -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so
RUN source devel/setup.bash

#Install rosbag for python3
#https://answers.ros.org/question/343260/install-of-the-rospyrosbag-python-libraries-standalone-not-on-ubuntu/
RUN pip install --extra-index-url https://rospypi.github.io/simple/ rospy rosbag

#Used for matplotlib to work in docker
#https://www.pyimagesearch.com/2015/08/24/resolved-matplotlib-figures-not-showing-up-or-displaying/
RUN sudo apt-get install -y tcl-dev tk-dev python-tk python3-tk