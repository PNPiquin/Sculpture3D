from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime
import time
import matplotlib.pyplot as plt

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
        if kinect.has_new_color_frame():
            frame = kinect.get_last_color_frame()
            print(frame)
            print(len(frame))
            break
        else:
            print('No frame')
            time.sleep(0.1)

    with open('frame.txt', 'w') as f:
        f.write(str(frame.tolist()))

    width = 1920
    height = 1080
    m = []
    # for k in range(3):
    #     offset = k * width * height
    #     m += [[[(frame[i * width + j + offset]) for j in range(width)] for i in range(height)]]
    m = [[(frame[(i * width + j) * 4]) for j in range(width)] for i in range(height)]
    plt.imshow(m)
    plt.show()
