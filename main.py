import skinSegAlg
import object2point
import pyrealsense2 as rs
import numpy as np
import cv2

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

while True:
    frames = pipeline.wait_for_frames() #获取摄像头的实时帧
    color_image,depth_image,depth_colormap = skinSegAlg.Align_version(frames,align,show_pic=0)
    hand_image = skinSegAlg.hand_YCbCr_ellipse(color_image)        

    # Stack both images horizontally 把两个图片水平拼在一起
    images = np.hstack((color_image, hand_image, depth_colormap))


    cloud = object2point.objectToPoint(hand_image, depth_image)
    cloud = np.array(cloud)
    # cloud = object2point.pointlesser(cloud)
    object2point.talker(cloud)        

    cv2.namedWindow(winname='RealSense',flags=cv2.WINDOW_AUTOSIZE) #设置视窗,flag为表示是否自动设置或调整窗口大小,WINDOW_AUTOSIZE即为自适应
    cv2.imshow('RealSense', images)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q') or key == 27:
        cv2.destroyAllWindows()
        break