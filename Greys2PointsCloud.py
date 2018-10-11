# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 09:27:27 2018

@author: tuoab
"""
import cv2


focalLength = 1
centerX = 75
centerY = 51
scalingFactor = 1.5


def generate_pointcloud(depth_img, output_file='cloudpoint.txt', image_file=None):
    points = []
    if image_file:
        color_img = cv2.imread(image_file)
    for v in range(len(depth_img)):
        for u in range(len(depth_img[0])):
            if depth_img[v][u] > 245:
                continue
            z = depth_img[v][u] / scalingFactor
            if image_file:
                points += [[u, v, z, color_img[v][u][0], color_img[v][u][1], color_img[v][u][2]]]
            else:
                points += [[u, v, z]]

    with open(output_file, 'w') as f:
        for i in range(len(points)):
            point = points[i]
            if image_file:
                for j in range(6):
                    f.write(str(point[j]) + " ")
            else:
                for j in range(3):
                    f.write(str(point[j]) + " ")
            f.write("\n")
            
    return points
