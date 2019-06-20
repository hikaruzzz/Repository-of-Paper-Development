'''
区域目标运动检测
关键字：
    cv2： 摄像头视频读取，运动物体框跟踪， 基于与初始背景图的差别得到二值图
    os：window控制，打开exe
    configparser：配置文件写入读取
'''
# coding=utf-8

import cv2
import os
import time
import numpy as np
import configparser


# config set
path = os.getcwd()
conf = configparser.ConfigParser()

# read and save config
while True:
    is_read = conf.read(filenames=path + r'\config_motion_detector')
    if is_read == []:
        # add section
        gadget_open_default = 'C:\Program Files\internet explorer\iexplore.exe'
        conf.add_section('detect_frame')
        conf.add_section('contours')
        conf.add_section('gadget_path')
        conf.add_section('binaryzation')
        conf.add_section('set')
        # set detect frame
        conf.set('detect_frame', 'detect_frame_vertical_1', '80')
        conf.set('detect_frame', 'detect_frame_vertical_2', '180')
        conf.set('detect_frame', 'detect_frame_horizon_1', '0')
        conf.set('detect_frame', 'detect_frame_horizon_2', '200')
        # for contours threshold
        conf.set('contours', 'contours_threshold', '1000')  # the size of contours covering motion object
        # for gadget path
        conf.set('gadget_path', 'gadget_path', gadget_open_default)
        conf.set('binaryzation', 'binaryzation_threshold', '8')
        conf.set('set','is_show_video','1')
        conf.write(open(path + r'\config_motion_detector', 'w+'))
        print("initialized config file")
    else:
        try:
            detect_frame_vertical_1 = int(conf.get('detect_frame','detect_frame_vertical_1'))
            detect_frame_vertical_2 = int(conf.get('detect_frame','detect_frame_vertical_2'))
            detect_frame_horizon_1 = int(conf.get('detect_frame','detect_frame_horizon_1'))
            detect_frame_horizon_2 = int(conf.get('detect_frame','detect_frame_horizon_2'))
            contours_threshold = int(conf.get('contours','contours_threshold'))
            binaryzation_threshold = int(conf.get('binaryzation','binaryzation_threshold'))
            is_show_video = int(conf.get('set','is_show_video'))
            gadget_open_default = conf.get('gadget_path','gadget_path')
            print("loaded config file")
            break
        except:
            pass

# gadget need opening
gadget_open = gadget_open_default

# capture video from camera
camera = cv2.VideoCapture(1) # 0:the first camera
time.sleep(0.5)
print("is camera opened:",camera.isOpened())

capture_size = (camera.get(cv2.CAP_PROP_FRAME_WIDTH),camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("camera size:",capture_size)

#  for inflation
es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,4))

background = None
calc_n = 0
is_open_gadget = 0
time_start = time.time()
time_flag = time_start
while True:
    # read video （every frame)
    grabbed,frame_lwpCV = camera.read()
    frame_lwpCV = frame_lwpCV[detect_frame_vertical_1:detect_frame_vertical_2,detect_frame_horizon_1:detect_frame_horizon_2]
    # preprocess, to gray ,gauss filter(gauss blur:reduce noise)
    gray_lwpCV = cv2.cvtColor(frame_lwpCV,cv2.COLOR_RGB2GRAY)
    gray_lwpCV = cv2.GaussianBlur(gray_lwpCV,(21,21),0)

    # first frame as background
    if background is None:
        background = gray_lwpCV
        continue

    # calculate differ between each frame and background,got a different map
    different_map_1 = cv2.absdiff(background,gray_lwpCV)
    # binaryzation
    different_map_2 = cv2.threshold(different_map_1,binaryzation_threshold,255,cv2.THRESH_BINARY)[1]
    # inflation
    #different_map_2 = cv2.dilate(different_map_2,es,iterations=2)

    if np.max(different_map_2) != 0:
        calc_n += 1
        print("warring! ! ! x",calc_n )
    if np.max(different_map_2) != 0 and is_open_gadget == 0:
        try:
            os.startfile(gadget_open)
        except:
            print("warring：wrong open path")
        is_open_gadget = 1

    # calculation contours
    # find contours !
    image,contours,hierarchy = cv2.findContours(different_map_2.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) < contours_threshold:
            # only show contours over threshold(reduce noise influence)
            continue
        # calculation rectangle boundary frame(show the green rectangle cover object[moving]）
        (x,y,w,h) = cv2.boundingRect(c)
        cv2.rectangle(frame_lwpCV,(x,y),(x+w,y+h),(0,255,0),2)


    # show
    if is_show_video == 1:
        cv2.imshow('contours:',frame_lwpCV)
        cv2.imshow('dis:',different_map_2) # show different map

    # exit judge
    if cv2.waitKey(1) == ord('q'):
        break

    # reset background , is_open_gadget
    if int(time.time() - time_start) % 5 == 0:
        background = gray_lwpCV

    if int(time.time() - time_start) % 10 == 0 and int(time.time()) - int(time_flag) > 1:
        is_open_gadget = 0
        time_flag = int(time.time())
    time.sleep(0.05)

camera.release()
cv2.destroyAllWindows()
