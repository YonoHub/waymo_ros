# waymo_ros

The package is made for ROS users who want to use Waymo Open Dataset with ROS.


# Installation and Usage

Create a workspace

mkdir -p catkin_ws/src
cd catkin_ws/src

Clone the repo to your workspace
git clone https://github.com/YonoHub/waymo_ros.git

You can find a setup.py file inside waymo_ros/scripts that you can use to install waymo_open_dataset dependancies:

cd waymo_ros/waymo_ros/scripts/
pip install .

Build the workspace:

cd ../../..
catkin_make


You can download Waymo dataset from here:
https://waymo.com/open/download/

Then extract the tar file and you'll have around 20 .tfrecord files.

Start the launch file:
source devel/setup.bash
roslaunch waymo_ros waymo_player.launch FilePath:="/path/to/file.tfrecord"
