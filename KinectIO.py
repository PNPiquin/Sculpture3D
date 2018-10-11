from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime


def get_depth_and_color_frame():
    """Fetch both color and depth image from the Kinect V2

    :return: (1D array, 1D array) (depth_frame, color_frame)
    """
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

    return depth_frame, color_frame
