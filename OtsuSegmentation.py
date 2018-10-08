import numpy as np
import KinectOutputProcessing as kop
import matplotlib.pyplot as plt


def otsu_segmentation(mat):
    min_, max_ = kop.matrix_extremum(mat)
    hist, bins = np.histogram(mat.flatten(), 256, [0, max_])
    plt.hist(mat.flatten(), 256, [0, max_])
    plt.show()

    p1_k = {}
    mean_k = {}

    for k in range(1, 256):
        p1_k[k] = 0
        mean_k[k] = 0
        for l in range(1, k+1):
            p1_k[k] += hist[l]
            mean_k[k] += l*hist[l]

    mean_global = mean_k[255]

    between_class_variances = {}
    for k in range(1, 256):
        between_class_variances[k] = (mean_global * p1_k[k] - mean_k[k])**2 / (p1_k[k] * (1 - p1_k[k]))

    k_max = 0
    var_max = 0
    for k in range(1, 256):
        if between_class_variances[k] > var_max:
            var_max = between_class_variances[k]
            k_max = k

    return k_max * max_ / 256