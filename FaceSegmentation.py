import matplotlib.pyplot as plt
import numpy as np


def face_segmentation(img, offset=5):
    height = len(img)
    width = len(img[0])
    hist = [[] for _ in range(height)]
    abs = [i for i in range(height)]
    for j in range(height):
        acu = 0
        for i in range(width):
            v = img[j][i]
            acu += v
        hist[j] = acu/width

    # plt.plot(abs, hist)
    # plt.show()

    start_index = int(height/2)
    max = 0
    index = start_index
    for j in range(start_index, height):
        if hist[j] > max:
            max = hist[j]
            index = j

    return min(index + 5, height)


def face_histogram_detection(img, otsu_thresh):
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

    # plt.hist(tmp_img.flatten(), 256, [0, otsu_thresh])
    # plt.show()
    return int(bins[k_1-1]), int(bins[k_2+1])
