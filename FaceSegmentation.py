import matplotlib.pyplot as plt


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
