# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 09:27:27 2018

@author: tuoab
"""


focalLength = 50
centerX = 75
centerY = 51
scalingFactor = 5


def generate_pointcloud(depth_img, output_file='cloudpoint.txt'):
    points = []    
    for v in range(len(depth_img)):
        for u in range(len(depth_img[0])):
            z = depth_img[v][u] / scalingFactor
#            if Z==0: continue
#            X = (u - centerX) * z / focalLength
#            Y = (v - centerY) * z / focalLength
            points += [[u, v, z]]

    with open(output_file, 'w') as f:
        for i in range(len(points)):
            point = points[i]
            for j in range(3):
                f.write(str(point[j]) + " ")
            f.write("\n")
            
    return points
