# coding=utf-8
import os
import copy
import cv2 as cv
import numpy as np
import open3d as o3d
import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import PointField
import yaml


filename= 'transforms.yaml'

def objectToPoint(hand_image, depth_image):
    cloud=[]
    # # 1.导入背景
    # color_background = cv.imread(os.path.join(abs_path, "images", "color_{}.png").format(background_num))
    # depth_background = cv.imread(os.path.join(abs_path, "images", "depth_{}.png").format(background_num),
    #                              cv.IMREAD_UNCHANGED).astype(np.int16)
    # # print(color_background)
    # #cv.imshow("background",color_background)

    # # 2.导入含有物体的图像
    # color_object = cv.imread(os.path.join(abs_path, "images", "color_{}.png").format(object_num))
    # depth_object = cv.imread(os.path.join(abs_path, "images", "depth_{}.png").format(object_num),
    #                          cv.IMREAD_UNCHANGED).astype(np.int16)
    # # cv.imshow("new_color",colorObject)
    # # print(color_object)

    # # 3.计算图像像素差值
    # abs_depth = np.abs(depth_object - depth_background)
    # threshold_image = np.zeros(abs_depth.shape).astype(np.uint8)
    # threshold_image[abs_depth > np.mean(abs_depth)] = 255
    # threshold_image[abs_depth < np.mean(abs_depth)] = 0
    # # cv.imshow("threshold",threshold_image)

    # # 4.开操作过滤噪声
    # kernel_size = 20
    # element = np.ones((kernel_size, kernel_size), np.uint8)
    # threshold_image = cv.morphologyEx(threshold_image, cv.MORPH_OPEN, kernel=element)
    # # cv.imshow("after process",threshold_image)

    # 5.提取结果，获取轮廓
    hand_image = cv.cvtColor(hand_image, cv.COLOR_BGR2GRAY)
    contours, hierarchy = cv.findContours(hand_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    mask_list = []
    
    for contour in contours:
        mask = np.zeros(hand_image.shape)
        cv.drawContours(mask, [contour], contourIdx=-1, color=255, thickness=-1)
        mask_list.append(copy.deepcopy(mask))
    # print(len(mask_list))
    # 6.利用内参矩阵获取分割后的点云
    camera_matrix = np.array(
        [616.536445, 0.000000, 324.960123, 0.000000, 617.225309, 237.878774, 0.000000, 0.000000, 1.000000]).reshape(3,3)
    cam_cx = 324.960123
    cam_cy = 237.878774
    cam_fx = 616.536445
    cam_fy = 617.225309
    pointcloud_list = []
    xmap = np.array([[j for i in range(640)] for j in range(480)])
    ymap = np.array([[i for i in range(640)] for j in range(480)])
    # print(xmap.shape)
    
    for mask in mask_list:
        choose = mask.flatten().nonzero()[0]

        depth_masked = depth_image.flatten()[choose][:, np.newaxis]
        x_masked = xmap.flatten()[choose][:, np.newaxis]
        y_masked = ymap.flatten()[choose][:, np.newaxis]

        pt2 = depth_masked
        pt0 = (y_masked - cam_cx) * pt2 / cam_fx
        pt1 = (x_masked - cam_cy) * pt2 / cam_fy
        cloud = np.concatenate((pt0, pt1, pt2), axis=1).astype(np.float32)
        # cloud = cloud.reshape(-1, 3)
    cloud = np.array(cloud)
    if cloud.ndim == 2:
        cloud = cloud[~(cloud==0).all(1)]
    # print("cloud type is",type(cloud))
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(cloud)
    # o3d.io.write_point_cloud(os.path.join(abs_path, "images", "objectPoint.pcd",cloud))
    # cloud = cloud/1000
    # print(mask_list)
    # cv.waitKey(0)
    
    return cloud



def talker(points, rate=30):

    pub = rospy.Publisher('object_topic', PointCloud2, queue_size=5)
    rospy.init_node('object_pointcloud_publisher_node', anonymous=True)
    rate = rospy.Rate(rate)
    # points = np.array(points)
    points = points / 1000
    # print(points.shape)
    # while not rospy.is_shutdown():

    msg = PointCloud2()
    msg.header.stamp = rospy.Time().now()
    msg.header.frame_id = "base"

    if len(points.shape) == 3:
        msg.height = points.shape[1]
        msg.width = points.shape[0]
    else:
        msg.height = 1
        msg.width = len(points)

    msg.fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1)]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = msg.point_step * points.shape[0]
    msg.is_dense = False
    msg.data = np.asarray(points, dtype=np.float32).tostring()

    pub.publish(msg)
    print("published...")
    rate.sleep()

def pointlesser(points,rate=30):
    lesspoints = points[0: -1: rate]

    return lesspoints

def radius_outlier(cloudpoint, nb_points = 10,radius = 0.2):

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(cloudpoint)
    #点云数据构造kd树
    pcd_tree = o3d.geometry.KDTreeFlann(cloud)
    #寻找每个点的邻近nb_points个数的点，并且计算他们之间的距离，如果距离大于radius，则舍弃该点，小于则保留
    di = []
    new_cloud = []
    for i in range(np.array(cloud.points).shape[0]):#遍历所有点
        [k,idx,_] = pcd_tree.search_knn_vector_3d(cloud.points[i],nb_points)
        #计算该点到每个点的欧式距离
        euc_distance = [ np.linalg.norm(np.array(cloud.points)[j] - np.array(cloud.points)[idx[0]])for j in np.array(idx)[1:]]
        is_less_than_radius = [j for j in euc_distance if j > radius ]
        if len(is_less_than_radius) == 0 :#所有距离都符合
            new_cloud.append(np.array(cloud.points)[i])
    new_cloud = np.array(new_cloud).reshape(-1,3)
    # new_pcd = o3d.geometry.PointCloud()
    # points = o3d.utility.Vector3dVector(np.array(new_cloud))
    # new_pcd.points = points
    # o3d.io.write_point_cloud('radius_deal_points.pcd',new_pcd,True)
    # o3d.visualization.draw_geometries([new_pcd])

    return new_cloud

def transform(points):

    with open(filename, 'r') as yaml_file:
        data = yaml.load(yaml_file, Loader=yaml.FullLoader)
    camToworld_matrix = np.array(data['H_cameraToworld']['data']).reshape(4, 4)
    worldTobase_matrix = np.array(data['H_worldToBase']['data']).reshape(4, 4)
    print("cameraMatrix",camToworld_matrix)
    print("matrix",worldTobase_matrix)
    camToworld_matrix[0,3] = camToworld_matrix[0,3]*1000
    camToworld_matrix[1,3] = camToworld_matrix[1,3]*1000
    camToworld_matrix[2,3] = camToworld_matrix[2,3]*1000
    worldTobase_matrix[0,3] = worldTobase_matrix[0,3]*1000
    worldTobase_matrix[1,3] = worldTobase_matrix[1,3]*1000
    worldTobase_matrix[2,3] = worldTobase_matrix[2,3]*1000
    # transformMatrix = np.array([-0.040899, 0.999043, -0.015473, -45.672, -0.998804, -0.041294, -0.026168, 16.5585, -0.026782, 0.014385, 0.999538, -700.447, 0.000000, 0.000000, 0.000000, 1.000000]).reshape(4,4)
    ones = np.ones(points.shape[0]).T
    # print(type(ones))
    ones = ones.reshape(points.shape[0],1)
    homo_points = np.hstack((points,ones))
    print(homo_points.shape)

    zfan = np.array([1,0,0,0,0,1,0,0,0,0,-1,0,0,0,0,1]).reshape(4,4)

    transformed_points = np.dot(homo_points,camToworld_matrix.T)
    transformed_points = np.dot(transformed_points,zfan.T)
    transformed_points = np.dot(transformed_points,worldTobase_matrix.T)
    transformed_points = np.delete(transformed_points,obj=3,axis=1)
    print("base",transformed_points[:3,:3])


    return transformed_points