import numpy as np
import cv2


def depth_array_to_matrix(depth_arr, width=512, height=424):
    depth_matrix = np.array([np.array([depth_arr[i * width + j] for j in range(width)]) for i in range(height)])
    return depth_matrix


def color_array_to_rgb_matrix(color_arr, width=1920, height=1080):
    color_matrix = []
    for h in range(height):
        color_matrix += [[]]
        for w in range(width):
            color_matrix[h] += [[color_arr[(h*width + w) * 4 + 2],
                                color_arr[(h*width + w) * 4 + 1],
                                color_arr[(h*width + w) * 4]]]
    return np.array(color_matrix)


def value_normalization(v, v_min=500, v_max=800):
    if v < v_min:
        return 255
    elif v > v_max:
        return 255
    else:
        return int(255 * (v-v_min) / (v_max - v_min))


def convolve_edge_detect(img):
    _filter = np.array([np.array([-1, -1, -1]), np.array([-1, 8, -1]), np.array([-1, -1, -1])])
    mat = cv2.filter2D(img, -1, _filter)
    return mat


def submatrix_mean(mat, i1, i2, j1, j2):
    arr = []
    for i in range(i1, i2):
        for j in range(j1, j2):
            v = mat[i][j]
            if v == 0:
                pass
            else:
                arr += [v]

    np_arr = np.array(arr)
    return np_arr.mean()


def submatrix_mean_std(mat, i1, i2, j1, j2):
    arr = []
    for i in range(i1, i2):
        for j in range(j1, j2):
            v = mat[i][j]
            if v == 0:
                pass
            else:
                arr += [v]

    np_arr = np.array(arr)
    return np_arr.mean(), np_arr.std(), np_arr.min(), np_arr.max(), np.quantile(np_arr, [0.1, 0.25, 0.5, 0.75, 0.8])


def matrix_quantile(mat):
    arr = []
    for i in range(0, len(mat)):
        for j in range(0, len(mat[0])):
            v = mat[i][j]
            if v == 0:
                pass
            else:
                arr += [v]

    np_arr = np.array(arr)
    return np.quantile(np_arr, [0.1, 0.70])


def matrix_extremum(mat):
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
    res = [[] for _ in range(y2 - y1)]
    for j in range(y1, y2):
        for i in range(x1, x2):
            res[j - y1] += [mat[j][i]]
    return np.array(res)
