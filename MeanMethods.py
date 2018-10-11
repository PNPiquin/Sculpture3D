from KinectOutputProcessing import submatrix_mean


def find_min_mean(mat, depth=3, width=512, height=424):
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