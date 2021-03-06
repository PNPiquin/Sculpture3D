import numpy as np
import cv2


def image_enhancement(img):
    """Double the width and the height of the input image by simple interpolation

    :param img: input image
    :return: the interpolated image
    """
    new_img = []
    height = len(img)
    width = len(img[0])
    for j in range(height-1):
        new_img += [[]]
        for i in range(width-1):
            new_img[2*j] += [img[j][i]]
            new_img[2*j] += [np.uint8((int(img[j][i]) + int(img[j][i+1])) / 2)]
        new_img[2*j] += [img[j][width-1], img[j][width-1]]

        new_img += [[]]
        for i in range(width-1):
            new_img[2*j+1] += [np.uint8((int(img[j][i]) + int(img[j+1][i])) / 2)]
            new_img[2*j+1] += [np.uint8((int(img[j][i]) + int(img[j][i+1]) + int(img[j+1][i]) + int(img[j+1][i+1])) / 4)]
        new_img[2*j+1] += [np.uint8((int(img[j][width-1]) + int(img[j+1][width-1])) / 2),
                           np.uint8((int(img[j][width-1]) + int(img[j+1][width-1])) / 2)]

    return np.array(new_img)


def image_opening(img, c=0.7, thresh=225):
    """Process an opening on the image to reconstruct the eyes

    :param img: input image
    :param c: (float) (0 < c < 1) weight to give to the mask image (img after opening)
    :param thresh: (int) background threshold
    :return: the processed image
    """
    er_img = cv2.dilate(img, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    open_img = cv2.morphologyEx(er_img, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)), iterations=1)
    new_img = img.copy()
    for j in range(len(img)):
        for i in range(len(img[0])):
            if img[j][i] > thresh:
                if open_img[j][i] < thresh:
                    new_img[j][i] = int(c * open_img[j][i] + (1 - c) * img[j][i])
    return new_img


def subtract_grad(img, thresh=40):
    """NOT USED

    :param img:
    :param thresh:
    :return:
    """
    grad = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    new_img = img.copy()
    for j in range(len(img)):
        for i in range(len(img[0])):
            if grad[j][i] > thresh:
                new_img[j][i] = min(255, img[j][i] + grad[j][i])

    return new_img
