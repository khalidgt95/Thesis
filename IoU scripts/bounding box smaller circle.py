# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 13:19:27 2019

@author: 061221
"""

import cv2
import numpy as np
import math

COLOR_BLUE = (255,0,0)
COLOR_GREEN = (0,255,0)
COLOR_RED = (0,0,255)

def display_image(image):
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return

img = np.zeros((512,512,3), np.uint8)


top_left_coordinate = (200,200)           ## x1, y1
bottom_right_coordinate = (370,450)       ## x2, y2
color = (0,255,0)
thickness = 1

origin_point_x = top_left_coordinate[0] + int((math.fabs(top_left_coordinate[0] - bottom_right_coordinate[0]) / 2) ) 
origin_point_y = top_left_coordinate[1] + int((math.fabs(top_left_coordinate[1] - bottom_right_coordinate[1]) / 2))
radius = int((math.fabs(top_left_coordinate[0] - bottom_right_coordinate[0]) / 2) )

major_axis = radius
minor_axis = int((math.fabs(top_left_coordinate[1] - bottom_right_coordinate[1]) / 2))

img = cv2.rectangle(img, top_left_coordinate, bottom_right_coordinate, color, thickness)



#circle_img = cv2.circle(img, (origin_point_x, origin_point_y), radius, COLOR_RED, 1)

ellipse_circle_img = cv2.ellipse(img, (origin_point_x, origin_point_y), (major_axis, minor_axis), 0,0,360, COLOR_BLUE, 1)

display_image(ellipse_circle_img)