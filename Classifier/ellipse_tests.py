#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 00:19:59 2019

@author: khalid
"""

import numpy as np
import cv2
import math

from keras_frcnn.simple_parser import get_data

def get_origin_points_for_ellipse_from_rectangle(coordinates):

    top_left_coordinate = (coordinates["x1"],coordinates["y1"])           ## x1, y1
    bottom_right_coordinate = (coordinates["x2"],coordinates["y2"])       ## x2, y2

    origin_point_x = top_left_coordinate[0] + int((math.fabs(top_left_coordinate[0] - bottom_right_coordinate[0]) / 2) )
    origin_point_y = top_left_coordinate[1] + int((math.fabs(top_left_coordinate[1] - bottom_right_coordinate[1]) / 2))

    radius = int((math.fabs(top_left_coordinate[0] - bottom_right_coordinate[0]) / 2) )

    major_axis = radius
    minor_axis = int((math.fabs(top_left_coordinate[1] - bottom_right_coordinate[1]) / 2))

    return [origin_point_x, origin_point_y, major_axis, minor_axis]


all_imgs, classes_count, class_mapping = get_data("annotate_ellipse.txt")

datalist = []

for image_num in range(5):

    img = cv2.imread(all_imgs[image_num]["filepath"])
    
    for bbox_num in range(len(all_imgs[image_num]["bboxes"])):
    
#        cords = get_origin_points_for_ellipse_from_rectangle(all_imgs[image_num]["bboxes"][bbox_num])
    
        cv2.ellipse(img, (all_imgs[image_num]["bboxes"][bbox_num]["x"], all_imgs[image_num]["bboxes"][bbox_num]["y"]), 
                      (all_imgs[image_num]["bboxes"][bbox_num]["x_axis"], all_imgs[image_num]["bboxes"][bbox_num]["y_axis"]), 0,0,360, (0,0,255), 1)
        
#        cv2.rectangle(img, (all_imgs[image_num]["bboxes"][bbox_num]["x"], all_imgs[image_num]["bboxes"][bbox_num]["y"]), 
#                      (all_imgs[image_num]["bboxes"][bbox_num]["x_axis"], all_imgs[image_num]["bboxes"][bbox_num]["y_axis"]), (0,0,255), 1)
#    
#    datalist.append(all_imgs[image_num]["filepath"]+","+str(cords[0])+","+str(cords[1])+","+str(cords[2])+","+str(cords[3])+","+all_imgs[image_num]["bboxes"][bbox_num]["class"])
    cv2.imwrite(str(image_num)+"_ellipse.png", img)

#    cv2.imshow("jhg", img)
#    cv2.waitKey()
#    cv2.destroyAllWindows()


## Code to export elliptical coordinates to csv file
#import pandas as pd
#import csv
#df = pd.DataFrame(datalist)
#df.to_csv("annotate_ellipse.txt", index=False, quoting=csv.QUOTE_NONE, header=None, sep=' ')    
    
    



