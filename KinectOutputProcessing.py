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
        return 255 * (v-v_min) / (v_max - v_min)


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
                arr += [5000]
            else:
                arr += [v]

    np_arr = np.array(arr)
    return np_arr.mean()


def fin_min_mean(mat, depth=3, width=512, height=424):
    i1 = 0
    i2 = height
    j1 = 0
    j2 = width
    for d in range(depth):
        min = -1
        i_step = int((i2 - i1) / 4)
        j_step = int((j2 - j1) / 4)
        tmp_i1 = 0
        tmp_j1 = 0
        for k in range(3):
            for l in range(3):
                mean = submatrix_mean(mat,
                                      i1 + k * i_step, i1 + (k+2) * i_step,
                                      j1 + l * j_step, j1 + (l+2) * j_step)
                if mean < min or min == -1:
                    min = mean
                    tmp_i1 = i1 + k * i_step
                    tmp_j1 = j1 + l * j_step

        i1 = tmp_i1
        i2 = tmp_i1 + 2 * i_step
        j1 = tmp_j1
        j2 = tmp_j1 + 2 * j_step

    return i1, i2, j1, j2
