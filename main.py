import matplotlib.pyplot as plt
import cv2
import pickle

import KinectOutputProcessing as kop
from OtsuSegmentation import otsu_segmentation
from KinectIO import get_depth_and_color_frame
from FaceRecognition import face_recognition

WITH_KINECT = False


if __name__ == '__main__':
    if WITH_KINECT:
        depth_frame, color_frame = get_depth_and_color_frame()

        # pickle.dump(depth_frame, open('depth_frame.pck', 'wb'))
        # pickle.dump(color_frame, open('color_frame.pck', 'wb'))
    else:
        depth_frame = pickle.load(open('depth_frame.pck', 'rb'))
        color_frame = pickle.load(open('color_frame.pck', 'rb'))

    # depth matrix creation
    m_depth = kop.depth_array_to_matrix(depth_frame)

    # RGB matrix creation
    m_color = kop.color_array_to_rgb_matrix(color_frame)

    x1, x2, y1, y2 = face_recognition(m_color=m_color)

    cv2.rectangle(m_depth, (x1, y1), (x2, y2), (255, 0, 0), 3)
    print('(x1, y1) = ({}, {})'.format(x1, y1))
    print('(x2, y2) = ({}, {})'.format(x2, y2))

    # plt.imshow(m_depth)
    # plt.show()

    m_depth_resized = kop.resize_matrix(m_depth, x1, x2, y1, y2)
    m_2 = m_depth_resized.copy()

    plt.imshow(m_depth_resized)
    plt.show()

    otsu_thresh = otsu_segmentation(m_depth_resized)
    print(otsu_thresh)

    quantiles = kop.matrix_quantile(m_depth_resized, max_value=otsu_thresh)
    print(quantiles)
    for i in range(len(m_depth_resized)):
        for j in range(len(m_depth_resized[0])):
            m_2[i][j] = kop.value_normalization(m_depth_resized[i][j], v_min=(quantiles[0]), v_max=(quantiles[1]))

    plt.imshow(m_2)
    plt.show()
