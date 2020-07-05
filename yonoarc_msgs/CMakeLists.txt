cmake_minimum_required(VERSION 2.8.3)
project(yonoarc_msgs)

find_package(catkin REQUIRED COMPONENTS message_generation std_msgs geometry_msgs)

add_message_files(
  DIRECTORY msg
  FILES
  Bool.msg
  BoolArray.msg
  Empty.msg
  Float32.msg
  Float32Array.msg
  Float64.msg
  Float64Array.msg
  Int16.msg
  Int16Array.msg
  Int32.msg
  Int32Array.msg
  Int64.msg
  Int64Array.msg
  Int8.msg
  Int8Array.msg
  ObjectBBoxI.msg
  ObjectBBoxArray.msg
  Label.msg
  LabelI.msg
  String.msg
  StringArray.msg
  UInt16.msg
  UInt16Array.msg
  UInt32.msg
  UInt32Array.msg
  UInt64.msg
  UInt64Array.msg
  UInt8.msg
  UInt8Array.msg
  VehicleControl.msg
  PointArray.msg
)

generate_messages(DEPENDENCIES std_msgs geometry_msgs)

catkin_package(CATKIN_DEPENDS message_runtime std_msgs geometry_msgs)