# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 09:27:27 2018

@author: tuoab
"""

import argparse
import sys
import os
from PIL import Image


img_file="depth_image1.jpg"
depth_file="depth_image1.jpg"
focalLength = 50
centerX = 75
centerY = 51
scalingFactor = 10

def generate_pointcloud(img_file,depth_file):

    image = Image.open(img_file)
    depth = Image.open(depth_file).convert('I')

    if depth.mode != "I":
        raise Exception("Depth image is not in intensity format")
    points = []    
    for v in range(image.size[1]):
        for u in range(image.size[0]):
            Z = depth.getpixel((u,v)) / scalingFactor
#            if Z==0: continue
#            X = (u - centerX) * Z / focalLength
#            Y = (v - centerY) * Z / focalLength
            points+=[[u,v,Z]]
            
    return points 
            
points=generate_pointcloud(img_file,depth_file)
f = open("cloudpoint.txt",'w')


for i in range(len(points)):
    point=points[i]
    for j in range (3):
        f.write(str(point[j])+" ")
    f.write("\n")
            

f.close()