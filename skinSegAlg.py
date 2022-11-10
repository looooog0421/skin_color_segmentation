#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import sys
from unicodedata import name
import pyrealsense2 as rs
import numpy as np
import cv2
import object2point
import open3d as o3d
# Configure depth and color streams 

def Align_version(frames,align,show_pic=0):
    # 对齐版本
    aligned_frames = align.process(frames)
    depth_frame_aligned = aligned_frames .get_depth_frame()
    color_frame_aligned = aligned_frames .get_color_frame()
    # if not depth_frame_aligned or not color_frame_aligned:
    #     continue
    color_image_aligned = np.asanyarray(color_frame_aligned.get_data())
    depth_image_aligned = np.asanyarray(depth_frame_aligned.get_data())
    
    # Apply colormap on depth image (image must be converted to 8-bit per pixel first) 在深度图上用颜色渲染
    # convertScaleAbs可以对src中每一个元素做
    depth_colormap_aligned = cv2.applyColorMap(cv2.convertScaleAbs(depth_image_aligned, alpha=0.05), cv2.COLORMAP_JET)
    images_aligned = np.hstack((color_image_aligned, depth_colormap_aligned))
    if show_pic:
        cv2.imshow('aligned_images', images_aligned)
    return color_image_aligned,depth_image_aligned,depth_colormap_aligned

def vibrance(img, amount):
    """
    this function is an approximated implementation for vibrance filter by Photoshop that increases the saturation of
    an image in a way that the increasing amount for the low saturated pixels is more than the increasing amount for
    pixels that are already saturated
    Parameters:
        img (ndarray): input image in HSV color space
        amount (int): increasing vibrance amount
    Returns:
         image in HSV color space after applying vibrance filter
    """
    amount = min(amount, 100)
    sat_increase = ((255 - img[:, :, 1]) / 255 * amount).astype(np.uint8)
    img[:, :, 1] += sat_increase
    return img

def brightenShadows(img, amount):
    """
    this function increases the brightness of the dark pixels of an image
    Parameters:
        img (ndarray): input image in HSV color space
        amount (int): increasing brightness amount
    Returns:
         image in HSV color space after applying brightness filter
    """
    amount = min(amount, 100)
    val_inc = ((255 - img[:, :, 2]) / 255 * amount).astype(np.uint8)
    img[:, :, 2] += val_inc
    return img


def hand_YCbCr_ellipse(frames):
    #肤色识别颜色参数
    ecl_x = 113
    ecl_y = 156
    leng_x = 24
    leng_y = 23
    ang = 43

    skinCrCbHist = np.zeros((256, 256), dtype=np.uint8)
    cv2.ellipse(skinCrCbHist, (ecl_x,ecl_y), (leng_x,leng_y), ang, 0.0, 360.0, (255,255,255), -1)
    framesYCrCb = cv2.cvtColor(frames, cv2.COLOR_BGR2YCrCb)
    (y, Cr, Cb) = cv2.split(framesYCrCb)
    (x, y) = framesYCrCb.shape[0:2]
    skin = np.zeros((x,y), dtype=np.uint8)
    skin = skinCrCbHist[Cr,Cb]
    # print(skin.shape)
    
    
    
    skin = cv2.bitwise_and(frames,frames, mask=skin)
    element =cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, element)

    # cv2.namedWindow("dst", cv2.WINDOW_NORMAL)
    # cv2.imshow("dst", output)
    # cv2.waitKey(0)

    return skin


if __name__ == "__main__":
    pipeline = rs.pipeline()
    config = rs.config()

    #初始化了两个数据流类型(深度图和彩色图)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)  
    config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)

    way = rs.stream.color
    align = rs.align(way)
    profile = pipeline.start(config)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    # print("scale:", depth_scale)
    # 深度比例系数为： 0.0010000000474974513

    #肤色识别颜色参数
    ecl_x = 113
    ecl_y = 156
    leng_x = 24
    leng_y = 23
    ang = 43

    # try:
    while True:
        frames = pipeline.wait_for_frames() #获取摄像头的实时帧
        color_image,depth_image,depth_colormap = Align_version(frames,align,show_pic=0)
        hand_image = hand_YCbCr_ellipse(color_image)        

        # Stack both images horizontally 把两个图片水平拼在一起
        images = np.hstack((color_image, hand_image, depth_colormap))

    
        # cloud = object2point.objectToPoint(hand_image, depth_image)
        # cloud = np.array(cloud)
        # # cloud = object2point.pointlesser(cloud)
        # object2point.talker(cloud)        

        cv2.namedWindow(winname='RealSense',flags=cv2.WINDOW_AUTOSIZE) #设置视窗,flag为表示是否自动设置或调整窗口大小,WINDOW_AUTOSIZE即为自适应
        cv2.imshow('RealSense', images)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break