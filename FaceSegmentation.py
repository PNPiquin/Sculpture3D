import matplotlib.pyplot as plt
import numpy as np


def face_segmentation(img, offset=5):
    """Search along the y-axis the limit between the face and the torso

    :param img: face image in which we want to cut off the torso
    :param offset: (in px) offset to lower the cutting point in the neck
    :return: (int) the index to cut the image
    """
    height = len(img)
    width = len(img[0])
    hist = [[] for _ in range(height)]

    for j in range(height):
        acu = 0
        for i in range(width):
            v = img[j][i]
            acu += v
        hist[j] = acu/width

    start_index = int(height/2)
    max = 0
    index = start_index
    for j in range(start_index, height):
        if hist[j] > max:
            max = hist[j]
            index = j

    # Plot instruction
    # abs = [i for i in range(height)]
    # plt.plot(abs, hist)
    # plt.axvline(x=index, color='r')
    # plt.show()

    return min(index + offset, height)


def face_histogram_detection(img, otsu_thresh):
    """Detect the range of pixel values on the face (on the only object in front)

    :param img: face img
    :param otsu_thresh: threshold obtained by an Otsu segmentation to consider only objects in the front
    :return: (int, int) range (v_min, v_max)
    """
    tmp_img = img.copy()
    height = len(img)
    width = len(img[0])
    for j in range(height):
        for i in range(width):
            if img[j][i] > otsu_thresh:
                tmp_img[j][i] = 0

    hist, bins = np.histogram(tmp_img.flatten(), 256, [0, otsu_thresh])
    hist[0] = 0

    k_max = 0
    hist_max = 0
    for k in range(256):
        if hist[k] > hist_max:
            hist_max = hist[k]
            k_max = k

    # Now we want the range of this part of the histogram
    k_1 = k_max
    for k in range(k_max, 0, -1):
        if hist[k] == 0:
            k_1 = k
            break

    k_2 = k_max
    for k in range(k_max, 255, 1):
        if hist[k] == 0:
            k_2 = k
            break

    # Plot instruction
    # plt.plot([i for i in range(256)], hist)
    # plt.axvline(x=k_1, color='r')
    # plt.axvline(x=k_2, color='r')
    # plt.show()
    return int(bins[k_1-1]), int(bins[k_2+1])
