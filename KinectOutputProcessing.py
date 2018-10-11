import numpy as np
import cv2


def depth_array_to_matrix(depth_arr, width=512, height=424):
    """Create a 2D matrix image from a 1D array

    :param depth_arr: 1D array
    :param width: depth camera width
    :param height: depth camera height
    :return: 2D matrix
    """
    depth_matrix = np.array([np.array([depth_arr[i * width + j] for j in range(width)]) for i in range(height)])
    return depth_matrix


def color_array_to_rgb_matrix(color_arr, width=1920, height=1080):
    """Create a '2D' BGR matrix image from a 1D array

    :param color_arr: 1D array ([r,g,b,a,r,g,b,a,r,g,b,a,...])
    :param width: rgb camera width
    :param height: rgb camera height
    :return: 3D tensor --> '2D' matrix of [B, G, R] pixels
    """
    color_matrix = []
    for h in range(height):
        color_matrix += [[]]
        for w in range(width):
            color_matrix[h] += [[color_arr[(h*width + w) * 4 + 2],
                                color_arr[(h*width + w) * 4 + 1],
                                color_arr[(h*width + w) * 4]]]
    return np.array(color_matrix)


def value_normalization(v, v_min=500, v_max=800):
    """map v from [v_min, v_max] to [0, 255]

    :param v: value to process
    :param v_min:
    :param v_max:
    :return: (int) value between 0 and 255
    """
    if v < v_min:
        return 255
    elif v > v_max:
        return 255
    else:
        return int(255 * (v-v_min) / (v_max - v_min))


def submatrix_mean_std(mat, i1, i2, j1, j2, max_value=10000):
    """Compute mean and standard deviation of pixel values in a submatrix of mat

    :param mat: input image
    :param i1: submatrix left coordinate
    :param i2: submatrix right coordinate
    :param j1: submatrix top coordinate
    :param j2: submatrix bottom coordinate
    :param max_value: ignore value above this threshold
    :return: (float, float) --> (submatrix_mean, submatrix_standard_deviation)
    """
    arr = []
    for i in range(i1, i2):
        for j in range(j1, j2):
            v = mat[i][j]
            if v == 0 or v > max_value:
                pass
            else:
                arr += [v]

    np_arr = np.array(arr)
    return np_arr.mean(), np_arr.std()


def matrix_extremum(mat):
    """Find mat pixel value extremum

    :param mat: input image
    :return: (int, int) --> (min, max)
    """
    arr = []
    for i in range(0, len(mat)):
        for j in range(0, len(mat[0])):
            v = mat[i][j]
            if v == 0:
                pass
            else:
                arr += [v]

    np_arr = np.array(arr)
    return np_arr.min(), np_arr.max()


def resize_matrix(mat, x1, x2, y1, y2):
    """Resize the matrix to the given coordinates

    :param mat: input matrix
    :param x1: submatrix left coordinate
    :param x2: submatrix right coordinate
    :param y1: submatrix top coordinate
    :param y2: submatrix bottom coordinate
    :return: resized matrix
    """
    res = [[] for _ in range(y2 - y1)]
    for j in range(y1, y2):
        for i in range(x1, x2):
            res[j - y1] += [mat[j][i]]
    return np.array(res)
