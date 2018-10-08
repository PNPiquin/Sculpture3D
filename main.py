from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime
import time
import matplotlib.pyplot as plt
import KinectOutputProcessing as kop
import cv2
import numpy as np
import pickle


if __name__ == '__main__':
    print('Coucou')
    kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth | PyKinectV2.FrameSourceTypes_Color)
    color_frame = []
    depth_frame = []
    has_depth = False
    has_color = False
    while True:
        if kinect.has_new_depth_frame() and not has_depth:
            depth_frame = kinect.get_last_depth_frame()
            has_depth = True
        if kinect.has_new_color_frame() and not has_color:
            color_frame = kinect.get_last_color_frame()
            has_color = True

        if has_depth and has_color:
            break

    pickle.dump(depth_frame, open('depth_frame.pck', 'wb'))
    pickle.dump(color_frame, open('color_frame.pck', 'wb'))

    # m = kop.color_array_to_rgb_matrix(frame)
    # frame_norm = [kop.value_normalization(v, 400, 1000) for v in depth_frame]
    m_depth = kop.depth_array_to_matrix(depth_frame)

    # RGB matrix creation
    m_color = kop.color_array_to_rgb_matrix(color_frame)

    face_model = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
    faces = face_model.detectMultiScale(m_color)

    # color img dimensions
    color_width = 1920
    color_height = 1080

    # depth image dimensions
    depth_width = 512
    depth_height = 424

    offset = 100
    norm_0 = -1
    x1_ = 0
    x2_ = 0
    y1_ = 0
    y2_ = 0

    for face in faces:
        x1 = int(((face[0] - offset) / color_width) * depth_width)
        y1 = int(((face[1] - offset) / color_height) * depth_height)

        x2 = int(((face[0] + face[2] + offset) / color_width) * depth_width)
        y2 = int(((face[1] + face[3] + offset) / color_height) * depth_height)

        norm = pow((x2 - x1), 2) + pow((y2 - y1), 2)
        if norm > norm_0:
            norm_0 = norm
            x1_ = x1
            x2_ = x2
            y1_ = y1
            y2_ = y2

    cv2.rectangle(m_depth, (x1_, y1_), (x2_, y2_), (255, 0, 0), 3)
    print('(x1, y1) = ({}, {})'.format(x1_, y1_))
    print('(x2, y2) = ({}, {})'.format(x2_, y2_))
    print(kop.submatrix_mean_std(m_depth, x1_, x2_, y1_, y2_))

    plt.imshow(m_depth)
    plt.show()

    m_depth_resized = kop.resize_matrix(m_depth, x1_, x2_, y1_, y2_)
    m_2 = m_depth_resized.copy()

    plt.imshow(m_depth_resized)
    plt.show()

    min_, max_ = kop.matrix_extremum(m_depth_resized)
    hist, bins = np.histogram(m_depth_resized.flatten(), 256, [0, max_])
    plt.hist(m_depth_resized.flatten(), 256, [0, max_], color='r')
    for i in range(50, 256):
        hist[i] = 0

    m3 = np.hstack((m_depth_resized, hist))
    plt.imshow(m3)
    plt.show()

    quantiles = kop.matrix_quantile(m_depth_resized)
    print(quantiles)
    for i in range(len(m_depth_resized)):
        for j in range(len(m_depth_resized[0])):
            m_2[i][j] = kop.value_normalization(m_depth_resized[i][j], v_min=(quantiles[0]), v_max=(quantiles[1]))

    plt.imshow(m_2)
    plt.show()
