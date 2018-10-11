import numpy as np


def cubist_grid(img, grid_size=10):
    height = len(img)
    width = len(img[0])
    cubist_height = (height // grid_size) * grid_size
    cubist_width = (width // grid_size) * grid_size

    # print('CUBIST GRID: (height, width) --> ({}, {})  || (c_height, c_width) --> ({}, {})'.format(
    #     height,
    #     width,
    #     cubist_height,
    #     cubist_width
    # ))

    cubist_img = np.zeros((cubist_height, cubist_width))
    for j in range(0, cubist_height, grid_size):
        for i in range(0, cubist_width, grid_size):

            # Here we will compute the mean on the [(j, i), (j + grid_size, i + grid_size)] square
            acu = 0
            for jj in range(0, grid_size):
                for ii in range(0, grid_size):
                    acu += img[j + jj][i + ii]
            acu = acu / (grid_size**2)

            # creation of a part of the cubist img
            for jj in range(0, grid_size):
                for ii in range(0, grid_size):
                    cubist_img[j + jj][i + ii] = acu

    return cubist_img


def cubist_grid_with_ramp(img, grid_size=20):
    height = len(img)
    width = len(img[0])
    cubist_height = (height // grid_size) * grid_size
    cubist_width = (width // grid_size) * grid_size

    a = 0
    b = 0
    if (height // grid_size) % 2 == 0:
        a = 1
    if (width // grid_size) % 2 == 0:
        b = 1

    cubist_img = np.zeros((cubist_height - a * grid_size, cubist_width - b * grid_size))
    for j in range(0, cubist_height, 2 * grid_size):
        for i in range(0, cubist_width, 2 * grid_size):

            # Here we will compute the mean on the [(j, i), (j + grid_size, i + grid_size)] square
            acu = 0
            for jj in range(0, grid_size):
                for ii in range(0, grid_size):
                    acu += img[j + jj][i + ii]
            acu = acu / (grid_size ** 2)

            # creation of a part of the cubist img
            for jj in range(0, grid_size):
                for ii in range(0, grid_size):
                    cubist_img[j + jj][i + ii] = acu

    # here we have created half the cubist image
    for j in range(grid_size, cubist_height - grid_size, 2 * grid_size):
        for i in range(0, cubist_width, 2 * grid_size):
            # We will interpolate values for this part with values from the top and the bottom part
            top_value = cubist_img[j - grid_size][i]
            bottom_value = cubist_img[j + grid_size][i]
            step = int((bottom_value - top_value) / grid_size)
            # print('INTERPOLATION --> (top, bottom, step) --> ({}, {}, {})'.format(
            #     top_value,
            #     bottom_value,
            #     step
            # ))
            for jj in range(0, grid_size):
                for ii in range(0, grid_size):
                    cubist_img[j + jj][i + ii] = top_value + jj * step

    for j in range(0, cubist_height, 2 * grid_size):
        for i in range(grid_size, cubist_width - grid_size, 2 * grid_size):
            # We will interpolate values for this part with values from the top and the bottom part
            left_value = cubist_img[j][i - grid_size]
            right_value = cubist_img[j][i + grid_size]
            step = int((right_value - left_value) / grid_size)
            for jj in range(0, grid_size):
                for ii in range(0, grid_size):
                    cubist_img[j + jj][i + ii] = left_value + ii * step

    for j in range(grid_size, cubist_height - grid_size, 2 * grid_size):
        for i in range(grid_size, cubist_width - grid_size, 2 * grid_size):
            # We will interpolate values for this part with values from the top and the bottom part
            top_left_value = cubist_img[j-grid_size][i - grid_size]
            top_right_value = cubist_img[j-grid_size][i + grid_size]
            bottom_left_value = cubist_img[j + grid_size][i - grid_size]
            bottom_right_value = cubist_img[j + grid_size][i + grid_size]
            h_top_step = int((top_right_value - top_left_value) / grid_size)
            h_bottom_step = int((bottom_right_value - bottom_left_value) / grid_size)
            for jj in range(0, grid_size):
                v_j_step = int((bottom_left_value + ii * h_bottom_step - (top_left_value + ii * h_top_step)) / grid_size)
                for ii in range(0, grid_size):
                    cubist_img[j + jj][i + ii] = top_left_value + ii * h_top_step + jj * v_j_step

    return cubist_img
