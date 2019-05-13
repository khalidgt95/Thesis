# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 20:24:14 2019

@author: Khalid
"""

import numpy as np
from matplotlib import pyplot as plt
import math

def plot_circles(circle1, circle2, IoU_value):
    
    fig = plt.gcf()
    ax = fig.gca()

    ax.add_artist(circle1)
    ax.add_artist(circle2)
    
    ax.set_title("IoU : " + str(IoU_value))
    
    return

def generate_circles(starting_point_circle1, starting_point_circle2, radius_circle1, radius_circle2):
            
    circle1_coordinates = starting_point_circle1
    circle2_coordinates = starting_point_circle2    
    
    circle1 = plt.Circle(circle1_coordinates, radius=radius_circle1, color='blue', clip_on=False,alpha=0.5)
    circle2 = plt.Circle(circle2_coordinates, radius=radius_circle2, color='g', clip_on=False,alpha=0.5)
    
    return circle1, circle2

def calculate_distance_along_x_axis(starting_point_circle1, starting_point_circle2):
    
    return math.fabs(starting_point_circle1[0]-starting_point_circle2[0])
    
def calculate_overlapping_area(radius_circle1, radius_circle2, starting_point_circle1
                               , starting_point_circle2):
    
    d = calculate_distance_along_x_axis(starting_point_circle1, starting_point_circle2)
    
    R = radius_circle1
    r = radius_circle2
    
    block1 = (r**2) * math.acos((d**2 + r**2 - R**2) / (2 * d * r))
    block2 = (R**2) * math.acos((d**2 + R**2 - r**2) / (2 * d * R))
    block3 = 0.5 * (math.sqrt((-d+r+R)*(d+r-R)*(d-r+R)*(d+r+R)))
    
    area = block1 + block2 - block3
    
    return area

def calculate_circle_area(radius):
    return (math.pi * radius**2)
    
def IoU(radius_circle1, radius_circle2, overlapping_area):
    
    denominator = calculate_circle_area(radius_circle1) + calculate_circle_area(radius_circle2) - overlapping_area    
                                       
    iou = overlapping_area / denominator
    
    return iou

if __name__=="__main__":
    
    RADIUS_CIRCLE1 = 0.3
    RADIUS_CIRCLE2 = 0.3
    
    STARTING_POINT_CIRCLE1 = (1,0)
    STARTING_POINT_CIRCLE2 = (0.1,0)
    
    circle1, circle2 = generate_circles(starting_point_circle1=STARTING_POINT_CIRCLE1, starting_point_circle2=STARTING_POINT_CIRCLE2,
                     radius_circle1=RADIUS_CIRCLE1, radius_circle2=RADIUS_CIRCLE2)    
    
    area = calculate_overlapping_area(RADIUS_CIRCLE1, RADIUS_CIRCLE2, STARTING_POINT_CIRCLE1, 
                                      STARTING_POINT_CIRCLE2)
    
    # When the circles completely overlap each other,
    # the overlapping area should be the same as the area for the individual circle
    #
    # When I checked with the same centers
    # This area is the same as when I compute manually
    #print(area)
    IoU_value = IoU(RADIUS_CIRCLE1, RADIUS_CIRCLE2, area)
    
    plot_circles(circle1, circle2, IoU_value)
    # 
    print("IoU : {} ".format(IoU_value))
    
    
    
    