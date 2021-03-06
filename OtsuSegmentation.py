import numpy as np
import KinectOutputProcessing as kop
import matplotlib.pyplot as plt


def otsu_segmentation(mat):
    """Process an Otsu segmentation to find the face value range in the depth image

    :param mat: input image
    :return: (float) --> threshold to separate the face from the background
    """
    min_, max_ = kop.matrix_extremum(mat)
    hist, bins = np.histogram(mat.flatten(), 256, [0, max_])
    # plt.hist(mat.flatten(), 256, [0, max_])
    # plt.show()

    # histogram normalization
    height = len(mat)
    width = len(mat[0])
    n_pixels = width * height
    histogram = []
    for k in range(256):
        histogram += [float(hist[k]) / n_pixels]

    hist[0] = 0

    p1_k = {}
    mean_k = {}

    for k in range(0, 256):
        p1_k[k] = 0
        mean_k[k] = 0
        for l in range(0, k+1):
            p1_k[k] += histogram[l]
            mean_k[k] += l*histogram[l]

    mean_global = mean_k[255]

    between_class_variances = {}
    for k in range(0, 256):
        if p1_k[k] != 0 and p1_k[k] != 1:
            between_class_variances[k] = float(pow(mean_global * p1_k[k] - mean_k[k], 2)) / (p1_k[k] * (1 - p1_k[k]))
        else:
            between_class_variances[k] = 0

    k_max = 0
    var_max = 0
    for k in range(0, 256):
        if between_class_variances[k] > var_max:
            var_max = between_class_variances[k]
            k_max = k

    # plt.plot([i for i in range(256)], hist)
    # plt.axvline(x=k_max, color='r')
    # plt.show()

    return k_max * max_ / 256
