# Self-supervised Machine Learning Pipeline for Obstacle Avoidance
The goal of this project is to develop a self-supervised training and obstacle 
avoidance algorithm, where a 2 dimensional image serves as input, whereas
a laser scanner provides the relevant ground truth necessary for training.
Self-supervision has the advantage, that no human labelling is necessary to gather
data for training. The project will show that this is possible with a simple camera
and a laser scanner. Other projects, which deal with similar approached, will be 
referred to. The goal is further, to implement as many pieces, of a machine learning 
pipeline, as autonomous as possible, where a robot can be started in a given environment,
gathering data, use that data for training and then operate on the deployed
model.

## Setup
Following link provides further information about how to properly set up this
project.

- [Setup](setup/README.md)

## Launch implementation
The source folder of this repository contains the implementation of the pipeline. As the pipeline is set up fully autonomous, the individual scripts might have to be adjusted if launched individually.

### Autonomous mode
Before launching the script, the catkin workspace can be built with the following commands:
```bash
cd ~/catkin_ws
catkin build
source ~/catkin_ws/devel/setup.bash
```
An environment can be launched with the following command:
```bash
rosrun gazebo_ros spawn_model -file /root/catkin_ws/src/ss20_lanz_2d_obstacle_avoidance/source/files/environments/env3/model.sdf -sdf -model model
```


To launch the pipeline a specific launch script is used. The
script is located under [Launch pipline script](source/files/) and can be started as follows:
```bash
cd /root/catkin_ws/src/ss20_lanz_2d_obstacle_avoidance/source/files
bash ./launch_pipeline.sh
```
The script will ask the user to provide input about which stage of the pipeline
should be started. It will furthermore request an input wether a new set of
bagfiles should be recorded or if an already created amount should be used.
Once the script is launched, a new pipeline index is created which will relate
all created files to this index to compare them with different runs.

The function "def script_location_setup()" defines how many bagfiles
are recorded. The default value for testing purposes is 5. The value can be
changed under ROS Scripts -> args.

### Individual mode
The following links provide further information about each part of the pipeline.

<span id="LC1" class="line" lang="plaintext">source</span><br/>
<span id="LC2" class="line" lang="plaintext">│  </span><br/>
<span id="LC3" class="line" lang="plaintext">└───</span><a href="source/files/README.md">files</a><br/>
<span id="LC4" class="line" lang="plaintext">│  </span><br/>
<span id="LC5" class="line" lang="plaintext">└───</span><a href="source/ss20_01_data_acquisition/README.md"> ss20_01_data_acquisition</a><br/>
<span id="LC6" class="line" lang="plaintext">│</span><br/>
<span id="LC7" class="line" lang="plaintext">└───</span><a href="source/ss20_02_feature_extraction/README.md"> ss20_02_feature_extraction</a><br/>
<span id="LC8" class="line" lang="plaintext">│</span><br/>
<span id="LC9" class="line" lang="plaintext">└───</span><a href="source/ss20_03_training_model/README.md"> ss20_03_training_model</a><br/>
<span id="LC10" class="line" lang="plaintext">│</span><br/>
<span id="LC11" class="line" lang="plaintext">└───</span><a href="source/ss20_04_testing_model/README.md"> ss20_04_testing_model</a><br/>
<span id="LC12" class="line" lang="plaintext">│</span><br/>
<span id="LC13" class="line" lang="plaintext">└───</span><a href="source/ss20_05_visualizing_model/README.md"> ss20_05_visualizing_model</a><br/>
<span id="LC14" class="line" lang="plaintext">│</span><br/>
<span id="LC15" class="line" lang="plaintext">└───</span><a href="source/ss20_06_visualizing_dataset/README.md"> ss20_06_visualizing_dataset</a><br/>
<span id="LC16" class="line" lang="plaintext">│</span><br/>
<span id="LC17" class="line" lang="plaintext">└───</span><a href="source/ss20_07_deploying_model/README.md"> ss20_07_deploying_model</a><br/>

## Results
Results are created automatically during each pipeline run and can be acessed
at the following link. Each markdown file represents one Model created.

- [Results](source/files/summary/)

## Build dependencies
- [Dockerfile](dockerfile/Dockerfile)

## Run dependencies
In order to properly run the image, it is required to have the **nvidia driver**,
the **nvidia toolkit** and the **nvidia runtime** set up.
installed properly.

## Nvidia driver setup
- [Nvidia driver](https://www.mvps.net/docs/install-nvidia-drivers-ubuntu-18-04-lts-bionic-beaver-linux/)
## Nvidia toolkit setup
- [Nvidia toolkit](https://github.com/NVIDIA/nvidia-docker)
## Nvidia runtime setup
- [Nvidia runtime](https://github.com/nvidia/nvidia-container-runtime)

## Authors
- Martin Lanz

## Problems and solutions
[Problems and solutions](https://fbe-gitlab.hs-weingarten.de/prj-iki-robotics/orga/robolab-wiki/wikis/Problems-And-Solutions)
