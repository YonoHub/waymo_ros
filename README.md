# waymo_ros
[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

The package is made for ROS users who want to use Waymo Open Dataset with ROS.


# Installation and Usage

Create a workspace
```sh
mkdir -p catkin_ws/src
cd catkin_ws/src
```

Clone the repo to your workspace
```sh
git clone https://github.com/YonoHub/waymo_ros.git
```

You can find a setup.py file inside waymo_ros/scripts that you can use to install waymo_open_dataset dependancies:
```sh
cd waymo_ros/waymo_ros/scripts/
pip install .
```
Build the workspace:
```sh
cd ../../..
catkin_make
```

You can download Waymo dataset from here:
https://waymo.com/open/download/

Then extract the tar file and you'll have around 20 .tfrecord files.

Start the launch file:
```sh
source devel/setup.bash
roslaunch waymo_ros waymo_player.launch FilePath:="/path/to/file.tfrecord"
```
