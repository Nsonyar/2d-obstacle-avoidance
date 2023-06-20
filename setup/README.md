# Setup
In order to set up this project, it is recommended to use the provided docker 
file to avoid installation and possible conflicts of dependencies on the current 
operating system. The following two sections describe of how to build the image 
and how to properly set up the package with setuptools.

## Building docker image
The project is mounted within a running docker container and can be used 
immediately after setting up the container with the provided 
[dockerfiles](../docker). Following instructions will describe of 
how to build the image.

After cloning the repository, change to the **/docker** folder and run the
following command in order to build the image:
```bash
sudo docker build -t ros_tiago_gpu:1.0 .
```
Once the image is built (approx 47 min), following command will start the container. 
The path of where the repository is located might have to be adjusted. The local 
volume will appear at **/catkin_ws/src/ss20_lanz_2d_obstacle_avoidance/**
```bash
docker run --gpus all -it -p 8879:8879 \
   --env="DISPLAY=$DISPLAY" \
   --env="QT_X11_NO_MITSHM=1" \
   --env=NVIDIA_DRIVER_CAPABILITIES="compute,video,utility,graphics" \
   --runtime="nvidia" \
   --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
   --volume="$HOME/.Xauthority:/root/.Xauthority"\
    -v $HOME/Bachelorthesis/ss20_lanz_2d_obstacle_avoidance:/root/catkin_ws/src/ss20_lanz_2d_obstacle_avoidance \
   --name ros ros_tiago_gpu:1.5
```

To check if GPU is supported, open a Python3 terminal from inside the docker
and run following commands:
```bash
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
```

**Important !**<br/>
To enable graphical user interfaces from inside docker, use the following
command from your host machine. (http://wiki.ros.org/docker/Tutorials/GUI)
This is necessary to have **Gazebo** and **matplotlib** running.
```bash
xhost +local:root
```
You can open another terminal with the following command:
```bash
docker exec -it ros bash
```

To check if gazebo is working, use following commands from inside
the container:
```bash
source ~/tiago_public_ws/devel/setup.bash
roslaunch tiago_gazebo tiago_gazebo.launch public_sim:=true robot:=steel
```

## Build package with setuptools
Once the container is up, following command needs to be run to install the
package and its dependencies from within the container:
```bash
cd /root/catkin_ws/src/ss20_lanz_2d_obstacle_avoidance
pip install -e .
```
This is basically making it easier to run scripts with sibling or parent imports
and avoid hacking the sys path for relative imports.

For further information see [setuptools](https://setuptools.readthedocs.io/en/latest/index.html)