#!/usr/bin/env python

import std_msgs.msg
import rospy
import cv2
from cv_bridge import CvBridge
import rospy
from yonoarc_msgs.msg import ObjectBBoxI, ObjectBBoxArray, LabelI
import time
import os
import tensorflow
import math
import numpy as np
import itertools
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs.msg import Image

from waymo_msgs.msg import ObjectBBoxArray3d, ObjectBBox3dI, LabelW


# print tensorflow.__version__
tensorflow.enable_eager_execution()

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset
import tf


waymo_to_labels = {
    0 : 'n02828884',
    1 : 'n02958343',
    2 : 'n00007846',
    3 : 'n06794110',
    4 : 'n02834778'
}


# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :

    assert(isRotationMatrix(R))
    
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    
    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])


def create_bbox_msg(ImageLabels):
    bbArray = ObjectBBoxArray()
    bbArray.header.stamp = rospy.Time.now()
    for x in range(0, len(ImageLabels)):
        bb = ObjectBBoxI()
        bb.id = 2
        bb.x = ImageLabels[x].box.center_x - 0.5 * ImageLabels[x].box.length 
        bb.y = ImageLabels[x].box.center_y - 0.5 * ImageLabels[x].box.width 
        bb.width = ImageLabels[x].box.length # bbox width
        bb.height = ImageLabels[x].box.width # bbox height
        ObjectLabel = LabelI(label_id=waymo_to_labels[ImageLabels[x].type], confidence=0.9)
        bb.labels = [ObjectLabel]
        bbArray.bboxes.append(bb)
    return bbArray

def create_3dbbox_msg(LaserLabels):
    bb3dArray = ObjectBBoxArray3d()
    bb3dArray.header.stamp = rospy.Time.now()
    for x in range(0, len(LaserLabels)):
        bb3d = ObjectBBox3dI()
        bb3d.id = 1
        bb3d.x = LaserLabels[x].box.center_x
        bb3d.y = LaserLabels[x].box.center_y
        bb3d.z = LaserLabels[x].box.center_z
        
        bb3d.width = LaserLabels[x].box.width
        bb3d.length = LaserLabels[x].box.length
        bb3d.height = LaserLabels[x].box.height
        bb3d.heading = LaserLabels[x].box.heading
        
        bb3d.speed_x = LaserLabels[x].metadata.speed_x
        bb3d.speed_y = LaserLabels[x].metadata.speed_y
        
        bb3d.acc_x = LaserLabels[x].metadata.accel_x
        bb3d.acc_y = LaserLabels[x].metadata.accel_y
        
        ObjectLabel = LabelW(label_id=waymo_to_labels[LaserLabels[x].type], confidence=0.9)
        bb3d.labels = [ObjectLabel]
        bb3dArray.bboxes3d.append(bb3d)
    return bb3dArray

rospy.loginfo("Waymo Dataset ROS Player")
rospy.init_node('WaymoDataset', anonymous=True)

PATH = rospy.get_param("DatasetPath")
CamF_pub = rospy.get_param("CamF_pub")
CamFL_pub = rospy.get_param("CamFL_pub")
CamSL_pub = rospy.get_param("CamSL_pub")
CamFR_pub = rospy.get_param("CamFR_pub")
CamSR_pub = rospy.get_param("CamSR_pub")

bboxes_pub = rospy.get_param("bboxes_pub")

LidF_pub = rospy.get_param("LidF_pub")
LidR_pub = rospy.get_param("LidR_pub")
LidSL_pub = rospy.get_param("LidSL_pub")
LidSR_pub = rospy.get_param("LidSR_pub")
LidT_pub = rospy.get_param("LidT_pub")

bboxes3d_pub = rospy.get_param("bboxes3d_pub")

pubcamF = rospy.Publisher('Fimage', Image, queue_size=10)
pubcamFL = rospy.Publisher('FLimage', Image, queue_size=10)
pubcamSL = rospy.Publisher('SLimage', Image, queue_size=10)
pubcamFR = rospy.Publisher('FRimage', Image, queue_size=10)
pubcamSR = rospy.Publisher('SRimage', Image, queue_size=10)


pubcambbF = rospy.Publisher('boundingboxesF', ObjectBBoxArray, queue_size=10)
pubcambbFL = rospy.Publisher('boundingboxesFL', ObjectBBoxArray, queue_size=10)
pubcambbSL = rospy.Publisher('boundingboxesSL', ObjectBBoxArray, queue_size=10)
pubcambbFR = rospy.Publisher('boundingboxesFR', ObjectBBoxArray, queue_size=10)
pubcambbSR = rospy.Publisher('boundingboxesSR', ObjectBBoxArray, queue_size=10)


pubLidarF = rospy.Publisher('flidar', PointCloud2, queue_size=10)
pubLidarR = rospy.Publisher('rlidar', PointCloud2, queue_size=10)
pubLidarSL = rospy.Publisher('sllidar', PointCloud2, queue_size=10)
pubLidarSR = rospy.Publisher('srlidar', PointCloud2, queue_size=10)
pubLidarT = rospy.Publisher('tlidar', PointCloud2, queue_size=10)


pubLaserLabels = rospy.Publisher('LaserLabels', ObjectBBoxArray3d, queue_size=10)
# print 'publishers declared'


bridge = CvBridge()

FImages = []
FImageLabels = []
FLImages = []
FLImageLabels = []
SLImages = []
SLImageLabels = []
FRImages = []
FRImageLabels = []
SRImages = []
SRImageLabels = []
lidars_frames = []
vehicle_poses = []
laser_labels = []


counter = 0

frames = []

rospy.loginfo("Processing Frames")
dataset = tensorflow.data.TFRecordDataset(PATH, compression_type='')
for data in dataset:
    frame = open_dataset.Frame()
    frame.ParseFromString(bytearray(data.numpy()))
    frames.append(frame)


frames_len = len(frames)



for i in range(0, len(frames)):
    frame_images = []
    for index, image in enumerate(frames[i].images):
        frame_images.append(tensorflow.image.decode_jpeg(image.image).numpy())
    
    FImages.append(frame_images[0])
    FLImages.append(frame_images[1])
    SLImages.append(frame_images[2])
    FRImages.append(frame_images[3])
    SRImages.append(frame_images[4])
    
    # labels
    FImageLabels.append(frames[i].camera_labels[0])
    FLImageLabels.append(frames[i].camera_labels[1])
    SLImageLabels.append(frames[i].camera_labels[2])
    FRImageLabels.append(frames[i].camera_labels[3])
    SRImageLabels.append(frames[i].camera_labels[4])
    
    # lidar points
    (range_images, camera_projections, range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frames[i])
    points, cp_points = frame_utils.convert_range_image_to_point_cloud(frames[i],range_images,camera_projections,range_image_top_pose)
    lidars_frames.append(points)
    
    vehicle_poses.append((np.array(frames[i].pose.transform)).reshape((4,4)))
    
    # laser labels (3d bounding boxes)
    laser_labels.append(frames[i].laser_labels)
    
    counter += 1
    rospy.loginfo("Processing Frame: " + str(counter))

    
    
    
TF_lidarF = (np.array(frames[0].context.laser_calibrations[0].extrinsic.transform)).reshape((4,4))
TF_lidarR = (np.array(frames[0].context.laser_calibrations[1].extrinsic.transform)).reshape((4,4))
TF_lidarSL = (np.array(frames[0].context.laser_calibrations[2].extrinsic.transform)).reshape((4,4))
TF_lidarSR = (np.array(frames[0].context.laser_calibrations[3].extrinsic.transform)).reshape((4,4))
TF_lidarT = (np.array(frames[0].context.laser_calibrations[4].extrinsic.transform)).reshape((4,4))

frames = []
# print 'data loaded'



freq = rospy.get_param("frequency")
rate = rospy.Rate(freq)

while not rospy.is_shutdown():
    for i in range(0, frames_len):
        rospy.loginfo("Publishing Frame: " + str(i))
        # front
        FImageMsg = Image()
        if CamF_pub is True:
            FImageMsg = bridge.cv2_to_imgmsg(FImages[i], encoding="passthrough")
            FImageMsg.header.stamp = rospy.Time.now()
            FImageMsg.header.frame_id = 'front_camera'

        # front left
        FLImageMsg = Image()
        if CamFL_pub is True:
            FLImageMsg = bridge.cv2_to_imgmsg(FLImages[i], encoding="passthrough")
            FLImageMsg.header.stamp = rospy.Time.now()
            FLImageMsg.header.frame_id = 'front_left_camera'

        # side left
        SLImageMsg = Image()
        if CamSL_pub is True:
            SLImageMsg = bridge.cv2_to_imgmsg(SLImages[i], encoding="passthrough")
            SLImageMsg.header.stamp = rospy.Time.now()
            SLImageMsg.header.frame_id = 'side_left_camera'

        # front right
        FRImageMsg = Image()
        if CamFR_pub is True:
            FRImageMsg = bridge.cv2_to_imgmsg(FRImages[i], encoding="passthrough")
            FRImageMsg.header.stamp = rospy.Time.now()
            FRImageMsg.header.frame_id = 'front_right_camera'

        # side right
        SRImageMsg = Image()
        if CamSR_pub is True:
            SRImageMsg = bridge.cv2_to_imgmsg(SRImages[i], encoding="passthrough")
            SRImageMsg.header.stamp = rospy.Time.now()
            SRImageMsg.header.frame_id = 'side_right_camera'
        
        
        bbArrayF = ObjectBBoxArray()
        bbArrayFL = ObjectBBoxArray()
        bbArraySL = ObjectBBoxArray()
        bbArrayFR = ObjectBBoxArray()
        bbArraySR = ObjectBBoxArray()
        
        bbArray3d = ObjectBBoxArray3d()
        
        
        if bboxes_pub is True:
#             print 'preparing bboxes'
            # front camera labels
            bbArrayF = create_bbox_msg(FImageLabels[i].labels)
            # front left camera labels
            bbArrayFL = create_bbox_msg(FLImageLabels[i].labels)
            # side left camera labels
            bbArraySL = create_bbox_msg(SLImageLabels[i].labels)
            # front right camera labels
            bbArrayFR = create_bbox_msg(FRImageLabels[i].labels)
            # side right camera labels
            bbArraySR = create_bbox_msg(SRImageLabels[i].labels)
        
        if bboxes3d_pub is True:
            bbArray3d = create_3dbbox_msg(laser_labels[i])
        
        # transform broadcasters
        
        # vehicle transform broadcaster
        rot_mat = vehicle_poses[i][0:3,0:3]
        [rx, ry, rz] = rotationMatrixToEulerAngles(rot_mat)
        X = vehicle_poses[i][0,3]
        Y = vehicle_poses[i][1,3]
        Z = vehicle_poses[i][2,3]
        
        br = tf.TransformBroadcaster()
        br.sendTransform((X, Y, Z),
                         tf.transformations.quaternion_from_euler(rx, ry, rz),
                         rospy.Time.now(),
                         "base_link",
                         "odom")
        
        rot_matF = TF_lidarF[0:3,0:3]
        [rxF, ryF, rzF] = rotationMatrixToEulerAngles(rot_matF)
        br = tf.TransformBroadcaster()
        br.sendTransform((TF_lidarF[0,3], TF_lidarF[1,3], TF_lidarF[2,3]),
                         tf.transformations.quaternion_from_euler(0, 0, 0),
                         rospy.Time.now(),
                         "front_laser",
                         "base_link")
        
        rot_matR = TF_lidarR[0:3,0:3]
        [rxR, ryR, rzR] = rotationMatrixToEulerAngles(rot_matR)
        br = tf.TransformBroadcaster()
        br.sendTransform((TF_lidarR[0,3], TF_lidarR[1,3], TF_lidarR[2,3]),
                         tf.transformations.quaternion_from_euler(0, 0, 0),
                         rospy.Time.now(),
                         "rear_laser",
                         "base_link")
        
        rot_matSL = TF_lidarSL[0:3,0:3]
        [rxSL, rySL, rzSL] = rotationMatrixToEulerAngles(rot_matSL)
        br = tf.TransformBroadcaster()
        br.sendTransform((TF_lidarSL[0,3], TF_lidarSL[1,3], TF_lidarSL[2,3]),
                         tf.transformations.quaternion_from_euler(0, 0, 0),
                         rospy.Time.now(),
                         "side_left_laser",
                         "base_link")
        
        rot_matSR = TF_lidarSR[0:3,0:3]
        [rxSR, rySR, rzSR] = rotationMatrixToEulerAngles(rot_matSR)
        br = tf.TransformBroadcaster()
        br.sendTransform((TF_lidarSR[0,3], TF_lidarSR[1,3], TF_lidarSR[2,3]),
                         tf.transformations.quaternion_from_euler(0, 0, 0),
                         rospy.Time.now(),
                         "side_right_laser",
                         "base_link")
        
        rot_matT = TF_lidarT[0:3,0:3]
        [rxT, ryT, rzT] = rotationMatrixToEulerAngles(rot_matT)
        br = tf.TransformBroadcaster()
        br.sendTransform((TF_lidarT[0,3], TF_lidarT[1,3], TF_lidarT[2,3]),
                         tf.transformations.quaternion_from_euler(0, 0, 0),
                         rospy.Time.now(),
                         "top_laser",
                         "base_link")
        
        
        # pointcloud2 msgs
        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                  PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1)]
        
        # front laser: 1
        FLidar = PointCloud2()
        if LidF_pub is True:
#             print 'preparing FLidar'
            header_pc2_F = std_msgs.msg.Header()
            header_pc2_F.stamp = rospy.Time.now()
            header_pc2_F.frame_id = 'front_laser'
            FLidar = pc2.create_cloud(header_pc2_F, fields, lidars_frames[i][1])
        
        
        
        
        # rear laser: 4
        RLidar = PointCloud2()
        if LidR_pub is True:
#             print 'preparing RLidar'
            header_pc2_R = std_msgs.msg.Header()
            header_pc2_R.stamp = rospy.Time.now()
            header_pc2_R.frame_id = 'rear_laser'
            RLidar = pc2.create_cloud(header_pc2_R, fields, lidars_frames[i][4])
        
        
        
        # side left laser: 2
        SLLidar = PointCloud2()
        if LidSL_pub is True:
#             print 'preparing SLLidar'
            header_pc2_SL = std_msgs.msg.Header()
            header_pc2_SL.stamp = rospy.Time.now()
            header_pc2_SL.frame_id = 'side_left_laser'
            SLLidar = pc2.create_cloud(header_pc2_SL, fields, lidars_frames[i][2])
        
        
        
        # side right laser: 3
        SRLidar = PointCloud2()
        if LidSR_pub is True:
#             print 'preparing SRLidar'
            header_pc2_SR = std_msgs.msg.Header()
            header_pc2_SR.stamp = rospy.Time.now()
            header_pc2_SR.frame_id = 'side_right_laser'
            SRLidar = pc2.create_cloud(header_pc2_SR, fields, lidars_frames[i][3])
        
        
        
        # top laser: 4
        TLidar = PointCloud2()
        if LidT_pub is True:
#             print 'preparing TLidar'
            header_pc2_T = std_msgs.msg.Header()
            header_pc2_T.stamp = rospy.Time.now()
            header_pc2_T.frame_id = 'top_laser'
            TLidar = pc2.create_cloud(header_pc2_T, fields, lidars_frames[i][0])
        
        
        if CamF_pub is True:
            pubcamF.publish(FImageMsg)
#             print 'Published FImageMsg'
        if bboxes_pub is True:
            pubcambbF.publish(bbArrayF)
#             print 'Published bbArrayF'
        
        if CamFL_pub is True:
            pubcamFL.publish(FLImageMsg)
#             print 'Published FLImageMsg'
        if bboxes_pub is True:
            pubcambbFL.publish(bbArrayFL)
#             print 'Published bbArrayFL'
        
        if CamSL_pub is True:
            pubcamSL.publish(SLImageMsg)
#             print 'Published SLImageMsg'
        if bboxes_pub is True:
            pubcambbSL.publish(bbArraySL)
#             print 'Published bbArraySL'
        
        if CamFR_pub is True:
            pubcamFR.publish(FRImageMsg)
#             print 'Published FRImageMsg'
        if bboxes_pub is True:
            pubcambbFR.publish(bbArrayFR)
#             print 'Published bbArrayFR'
        
        if CamSR_pub is True:
            pubcamSR.publish(SRImageMsg)
#             print 'Published SRImageMsg'
        if bboxes_pub is True:
            pubcambbSR.publish(bbArraySR)
#             print 'Published bbArraySR'
        
        
        
        # publishing Lidar Pointclouds
        if LidF_pub is True:
            pubLidarF.publish(FLidar)
#             print 'Published FLidar'
        
        if LidR_pub is True:
            pubLidarR.publish(RLidar)
#             print 'Published RLidar'
        
        if LidSL_pub is True:
            pubLidarSL.publish(SLLidar)
#             print 'Published SLLidar'
        
        if LidSR_pub is True:
            pubLidarSR.publish(SRLidar)
#             print 'Published SRLidar'
        
        if LidT_pub is True:
            pubLidarT.publish(TLidar)
#             print 'Published TLidar'
            
        if bboxes3d_pub is True:
            pubLaserLabels.publish(bbArray3d)
        
        rate.sleep() 
        
    Repeat = rospy.get_param("Repeat")
    if Repeat == False:
        break
        
        


    