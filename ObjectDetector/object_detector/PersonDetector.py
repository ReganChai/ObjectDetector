##  -*- coding = UTF-8 -*-
##
##  加载 Caffe 框架以及训练好的神经网络进行行人检测
##  可设置删选条件检测其他类型，该模型共包含 20 种可检测类型
##
##  Created on:  2018年4月26
##      Author: Regan_Chai
##      E-Mail: regan_chai@163.com
##

import numpy as np
import argparse

try:
    import cv2 as cv
except ImportError:
    raise ImportError('Can\'t find OpenCV Python module. If you\'ve built it from sources without installation, '
                      'configure environment variable PYTHONPATH to "opencv_build_dir/lib" directory (with "python3" subdirectory if required)')

from cv2 import dnn


prototxt = 'object_detector/MobileNetSSD_deploy.prototxt'    # 调用.caffemodel时的测试网络文件
caffemodel = 'object_detector/MobileNetSSD_deploy.caffemodel'  #包含实际图层权重的.caffemodel文件

video = "E:/视频库/标准库/camera4.mov"
image_file = "face01.jpg"

inWidth = 300
inHeight = 300
WHRatio = inWidth / float(inHeight)
inScaleFactor = 0.007843
meanVal = 127.5


def person_detector():
    net = dnn.readNetFromCaffe(prototxt, caffemodel)
    swapRB = False
    classNames = { 0: 'background',
            1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
            5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
            10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
            14: 'motorbike', 15: 'person', 16: 'pottedplant',
            17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor' }
   
    isFrame =  True  #True  #False
    if isFrame:
        cap = cv.VideoCapture(0)  #video
    else:
        frame = cv.imread(image_file)
        
    while True:
        if isFrame:
            ret, frame = cap.read()
        
        blob = dnn.blobFromImage(frame, inScaleFactor, (inWidth, inHeight), (meanVal, meanVal, meanVal), swapRB)
        net.setInput(blob)
        detections = net.forward()
        print(detections)
        cols = frame.shape[1]
        rows = frame.shape[0]

        if cols / float(rows) > WHRatio:
            cropSize = (int(rows * WHRatio), rows)
        else:
            cropSize = (cols, int(cols / WHRatio))

        y1 = int((rows - cropSize[1]) / 2)
        y2 = y1 + cropSize[1]
        x1 = int((cols - cropSize[0]) / 2)
        x2 = x1 + cropSize[0]
        frame = frame[y1:y2, x1:x2]

        cols = frame.shape[1]
        rows = frame.shape[0]

        confThreshold = 0.6   #置信度阈值
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > confThreshold:
                class_id = int(detections[0, 0, i, 1])
                
                if classNames[class_id] is 'person': #只识别行人，并进行标记

                    xLeftBottom = int(detections[0, 0, i, 3] * cols)
                    yLeftBottom = int(detections[0, 0, i, 4] * rows)
                    xRightTop   = int(detections[0, 0, i, 5] * cols)
                    yRightTop   = int(detections[0, 0, i, 6] * rows)
                    cv.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop), (0, 0, 255), 3)

                    label = classNames[class_id] + ": " + str(confidence)
                    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                    yLeftBottom = max(yLeftBottom, labelSize[1])
                    cv.rectangle(frame, (xLeftBottom, yLeftBottom - labelSize[1]),
                                        (xLeftBottom + labelSize[0], yLeftBottom + baseLine),
                                        (255, 255, 255), cv.FILLED)
                    cv.putText(frame, label, (xLeftBottom, yLeftBottom), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        cv.imshow("detections", frame)
        if cv.waitKey(3) >= 0:
            break

if __name__ == "__main__":
    person_detector()

