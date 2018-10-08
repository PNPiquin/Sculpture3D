from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime
import time
import matplotlib.pyplot as plt
import KinectOutputProcessing as kop


def norm(x):
    if x < 550:
        return 550
    elif x > 750:
        return 750
    else:
        return x


if __name__ == '__main__':
    print('Coucou')
    kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth | PyKinectV2.FrameSourceTypes_Color)
    frame = []
    while True:
        if kinect.has_new_depth_frame():
            frame = kinect.get_last_depth_frame()
            print(frame)
            print(len(frame))
            break
        else:
            print('No frame')
            time.sleep(0.1)

    # m = kop.color_array_to_rgb_matrix(frame)
    frame_norm = [kop.value_normalization(v, 620, 750) for v in frame]
    m = kop.depth_array_to_matrix(frame_norm)

    plt.imshow(m)
    plt.show()
