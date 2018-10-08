from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime
import time
import matplotlib.pyplot as plt
import KinectOutputProcessing as kop
import cv2


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

    # m = kop.color_array_to_rgb_matrix(frame)
    # frame_norm = [kop.value_normalization(v, 600, 900) for v in frame]
    m_depth = kop.depth_array_to_matrix(depth_frame)
    m_color = kop.color_array_to_rgb_matrix(color_frame)

    # m = kop.convolve_edge_detect(m)

    plt.imshow(m_depth)
    plt.show()

    plt.imshow(m_color)
    cv2.imwrite('pnp.jpg', m_color)
    
    plt.show()
