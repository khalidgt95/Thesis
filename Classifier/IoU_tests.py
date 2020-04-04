#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 00:50:53 2019

@author: khalid
"""

import numpy as np
from keras_frcnn import config, data_generators

C = config.Config()
downscale = float(C.rpn_stride)
anchor_sizes = C.anchor_box_scales
anchor_ratios = C.anchor_box_ratios
num_anchors = len(anchor_sizes) * len(anchor_ratios)
    
from shapely.geometry.point import Point
from matplotlib.patches import Polygon
from shapely import affinity

output_width = 50
output_height = 37

downscale = 16


resized_width = 800
resized_height = 600

def create_ellipse(center, lengths, angle=0):
    circ = Point(center).buffer(1)
    ell = affinity.scale(circ, int(lengths[0]), int(lengths[1]))
    ## Not used right now but will be in the future
    #ellr = affinity.rotate(ell, angle)
    return ell

## Here the rotation will become a paramater also in the next round
def iou(a, b):

    ellipse1 = create_ellipse((a[0], a[1]), (a[2], a[3]), 0)
    ellipse2 = create_ellipse((b[0], b[1]), (b[2], b[3]), 0)
    
    intersection = ellipse1.intersection(ellipse2)
    
    denominator = ellipse1.area + ellipse2.area - intersection.area
    
    iou = intersection.area / denominator
    
    print('area of intersection:',intersection.area)    
    
    return iou

## Unit tests
iou([113, 325, 90, 96],[114, 325, 90, 96])

for anchor_size_idx in range(len(anchor_sizes)):
    for anchor_ratio_idx in range(3):
        anchor_x = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][0]
        anchor_y = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][1]
        print(anchor_x/16, anchor_y/16)
        
        for ix in range(output_width):
            # x-co ordinates of the current anchor box
            x1_anc = downscale * (ix + 0.5) - anchor_x / 2
            x2_anc = downscale * (ix + 0.5) + anchor_x / 2
            
            center_x = x1_anc + ((x1_anc + x2_anc) / 2)
            x_axis = (x1_anc + x2_anc) / 2
            # ignore boxes that go across image boundaries
            if x1_anc < 0 or x2_anc > resized_width:
                continue

            for jy in range(output_height):

                # y-coordinates of the current anchor box
                y1_anc = downscale * (jy + 0.5) - anchor_y / 2
                y2_anc = downscale * (jy + 0.5) + anchor_y / 2

                center_y = y1_anc + ((y1_anc + y2_anc) / 2)
                y_axis = (y1_anc + y2_anc) / 2
                # ignore boxes that go across image boundaries
                if y1_anc < 0 or y2_anc > resized_height:
                    continue
                
                print(x1_anc, y1_anc, x2_anc, y2_anc)