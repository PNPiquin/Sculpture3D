import matplotlib.pyplot as plt
import cv2
import pickle
import numpy as np
import os

import KinectOutputProcessing as kop
from OtsuSegmentation import otsu_segmentation
from KinectIO import get_depth_and_color_frame
from FaceRecognition import face_recognition
from ImageEnhancement import image_enhancement, image_opening
from Greys2PointsCloud import generate_pointcloud
from FaceSegmentation import face_segmentation, face_histogram_detection
from StyleEffects import cubist_grid, cubist_grid_with_ramp

WITH_KINECT = False
SAVE_RAW_IMAGES = False
SAVE_TMP_IMAGES = True

presentation_path = os.path.join('.', 'Presentation')


if __name__ == '__main__':
    if WITH_KINECT:
        depth_frame, color_frame = get_depth_and_color_frame()

        if SAVE_RAW_IMAGES:
            pickle.dump(depth_frame, open('depth_frame_chloe.pck', 'wb'))
            pickle.dump(color_frame, open('color_frame_chloe.pck', 'wb'))
    else:
        depth_frame = pickle.load(open('depth_frame.pck', 'rb'))
        color_frame = pickle.load(open('color_frame.pck', 'rb'))

    # depth matrix creation
    m_depth = kop.depth_array_to_matrix(depth_frame)
    if SAVE_TMP_IMAGES:
        cv2.imwrite(os.path.join(presentation_path, 'depth_image.jpg'), m_depth)

    # RGB matrix creation
    m_color = kop.color_array_to_rgb_matrix(color_frame)
    if SAVE_TMP_IMAGES:
        cv2.imwrite(os.path.join(presentation_path, 'color_image.jpg'), m_color[...,::-1])

    x1, x2, y1, y2 = face_recognition(m_color=m_color)

    m_depth_resized = kop.resize_matrix(m_depth, x1, x2, y1, y2)
    m_2 = m_depth_resized.copy()
    m_3 = m_depth_resized.copy()

    if SAVE_TMP_IMAGES:
        cv2.imwrite(os.path.join(presentation_path, 'depth_resized.jpg'), m_depth_resized)

    # plt.imshow(m_depth_resized, cmap='Greys')
    # plt.show()

    otsu_thresh = otsu_segmentation(m_depth_resized)
    otsu_thresh = min(otsu_thresh, 1500)
    print(otsu_thresh)

    # This quantile method was not robust enough to be used
    #
    # quantiles = kop.matrix_quantile(m_depth_resized, max_value=otsu_thresh)
    # print(quantiles)
    #
    # for i in range(len(m_depth_resized)):
    #     for j in range(len(m_depth_resized[0])):
    #         m_2[i][j] = kop.value_normalization(m_depth_resized[i][j], v_min=(quantiles[0]), v_max=(quantiles[1]))
    #
    # fig = plt.figure(figsize=(1, 2))
    # fig.add_subplot(1, 2, 1)
    # plt.imshow(m_2, cmap='Greys')
    # cv2.imwrite('depth_face.jpg', m_2)

    v_min, v_max = face_histogram_detection(m_depth_resized, otsu_thresh)

    mean, std = kop.submatrix_mean_std(m_depth_resized, 0, y2 - y1, 0, x2 - x1, otsu_thresh)
    alpha = 3
    if v_min < (mean - alpha * std):
        v_min = mean - alpha * std
        print('WARNING: v_min is using std')
    if v_max > (mean + alpha * std):
        v_max = mean + alpha * std
        print('WARNING: v_max is using std')
    print(mean)
    print(std)
    for i in range(len(m_depth_resized)):
        for j in range(len(m_depth_resized[0])):
            m_3[i][j] = kop.value_normalization(m_depth_resized[i][j],
                                                v_min=v_min,
                                                v_max=v_max)

    m_4 = image_opening(m_3)
    if SAVE_TMP_IMAGES:
        cv2.imwrite(os.path.join(presentation_path, 'depth_face_without_opening.jpg'), m_3)
        cv2.imwrite(os.path.join(presentation_path, 'depth_face.jpg'), m_4)

    cut_index = face_segmentation(m_4)
    m_7 = kop.resize_matrix(m_4, 0, len(m_4[0]), 0, cut_index)

    m_6_ = cv2.bilateralFilter(m_7.astype(np.uint8), 5, 120, 80)
    m_6 = image_enhancement(image_enhancement(m_6_))

    if SAVE_TMP_IMAGES:
        cv2.imwrite(os.path.join(presentation_path, 'cut_depth_face.jpg'), m_7)
        cv2.imwrite(os.path.join(presentation_path, 'final_depth_face.jpg'), m_6)

    # plotting some images to monitor the process
    fig = plt.figure(figsize=(1, 3))
    fig.add_subplot(1, 3, 1)
    plt.imshow(m_7, cmap='Greys')
    fig.add_subplot(1, 3, 2)
    plt.imshow(m_6, cmap='Greys')

    m_cubist = cubist_grid_with_ramp(m_6, grid_size=15)

    fig.add_subplot(1, 3, 3)
    plt.imshow(m_cubist, cmap='Greys')
    plt.show()

    generate_pointcloud(m_6, 'cloud_bilateral.txt')
    generate_pointcloud(m_cubist, 'cloud_cubist.txt')
