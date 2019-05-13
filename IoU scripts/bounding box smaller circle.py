# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 20:24:14 2019

@author: Khalid
"""

import numpy as np
from matplotlib import pyplot as plt
import math

def plot_circles(circle1, circle2, IoU_value):
    
    """
    This function plots the circles and shows them to screen
    
    Parameters:
    -----------
    
    circle1: Circle object of matplotlib
    circle2: Circle object of matplotlib
    IoU_value: IoU value
    """
    
    fig = plt.gcf()
    ax = fig.gca()

    ax.add_artist(circle1)
    ax.add_artist(circle2)
    
    ax.set_title("IoU : " + str(IoU_value))
    
    return

def generate_circles(starting_point_circle1, starting_point_circle2, radius_circle1, radius_circle2):
            
    """
    This function generates the circles and calls the function to plot them
    
    Parameters:
    -----------
    
    starting_point_circle1:
    starting_point_circle1:
    radius_circle1:
    radius_circle2:
        
    """
    circle1_coordinates = starting_point_circle1
    circle2_coordinates = starting_point_circle2    
    
    circle1 = plt.Circle(circle1_coordinates, radius=radius_circle1, color='blue', clip_on=False,alpha=0.5)
    circle2 = plt.Circle(circle2_coordinates, radius=radius_circle2, color='g', clip_on=False,alpha=0.5)    
    
    return circle1, circle2

def calculate_distance_along_x_axis(starting_point_circle1, starting_point_circle2):
    
    ## Euclidean distance 
    distance = math.sqrt((starting_point_circle1[0]-starting_point_circle2[0])**2 + 
               (starting_point_circle1[1]-starting_point_circle2[1])**2)
    
    return distance
    
def calculate_overlapping_area(radius_circle1, radius_circle2, starting_point_circle1
                               , starting_point_circle2):
    
    d = calculate_distance_along_x_axis(starting_point_circle1, starting_point_circle2)
   
    ## When the distance is greater than the sum of the radius of two circles, 
    ## then the overlapping area becomes zero
    ## This case is for when the two circles do not overlap at all
    if d > (radius_circle1 + radius_circle2):
        print("distance : {} is greater than sum of radii : {}".format(d, radius_circle1+radius_circle2))
        return 0.0
    
    
    ## This case is for when the circles are of unequal length. If the circle lies inside other circle, 
    ## then do this
    if (d + min(radius_circle1, radius_circle2)) < max(radius_circle1, radius_circle2):
        print("One circle lies inside the other circle completely")
        return calculate_circle_area(min(radius_circle1, radius_circle2))
    
    print("distance : {}".format(d))
    R = radius_circle1
    r = radius_circle2
    
    print("block 1 {}".format((d**2 + r**2 - R**2) / (2 * d * r)))
    print("block 2 {}".format((d**2 + R**2 - r**2) / (2 * d * R)))
    ## Have to perform clipping here so that it doesnt exceed the bounds of inverse cos
    block1 = (r**2) * math.acos((d**2 + r**2 - R**2) / (2 * d * r))
    print(((-d+r+R)*(d+r-R)*(d-r+R)*(d+r+R)))
    block2 = (R**2) * math.acos((d**2 + R**2 - r**2) / (2 * d * R))
    block3 = 0.5 * (math.sqrt((-d+r+R)*(d+r-R)*(d-r+R)*(d+r+R)))
    
    print(block3)
    area = block1 + block2 - block3
    
    return area

def calculate_circle_area(radius):
    return (math.pi * radius**2)
    
def IoU(radius_circle1, radius_circle2, overlapping_area):
    
    denominator = calculate_circle_area(radius_circle1) + calculate_circle_area(radius_circle2) - overlapping_area    
                                       
    iou = overlapping_area / denominator
    
    return iou

if __name__=="__main__":
    
    RADIUS_CIRCLE1 = 0.2
    RADIUS_CIRCLE2 = 0.2

    STARTING_POINT_CIRCLE1 = (0.2,0.5)
    STARTING_POINT_CIRCLE2 = (0.7,0.5)
    
    circle1, circle2 = generate_circles(starting_point_circle1=STARTING_POINT_CIRCLE1, starting_point_circle2=STARTING_POINT_CIRCLE2,
                     radius_circle1=RADIUS_CIRCLE1, radius_circle2=RADIUS_CIRCLE2)    
    
    area = calculate_overlapping_area(RADIUS_CIRCLE1, RADIUS_CIRCLE2, STARTING_POINT_CIRCLE1, 
                                      STARTING_POINT_CIRCLE2)
    
    # When the circles completely overlap each other,
    # the overlapping area should be the same as the area for the individual circle
    #
    # When I checked with the same centers
    # This area is the same as when I compute manually
    print("Area of overlapping region : {}".format(area))
    
    iou = IoU(RADIUS_CIRCLE1, RADIUS_CIRCLE2, area)
    
    plot_circles(circle1, circle2, iou)
    
    print("IoU : {}".format(iou))
    
    # TODOs:
    # Discuss about the possible cases where the circle are of different sizes and engulf one another
    # Next steps