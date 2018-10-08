import cv2


def face_recognition(m_color):
    face_model = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
    faces = face_model.detectMultiScale(m_color)

    # color img dimensions
    color_width = 1920
    color_height = 1080

    # depth image dimensions
    depth_width = 512
    depth_height = 424

    offset = 100
    norm_0 = -1
    x1_ = 0
    x2_ = 0
    y1_ = 0
    y2_ = 0

    for face in faces:
        x1 = int(((face[0] - offset) / color_width) * depth_width)
        y1 = int(((face[1] - offset) / color_height) * depth_height)

        x2 = int(((face[0] + face[2] + offset) / color_width) * depth_width)
        y2 = int(((face[1] + face[3] + offset) / color_height) * depth_height)

        norm = pow((x2 - x1), 2) + pow((y2 - y1), 2)
        if norm > norm_0:
            norm_0 = norm
            x1_ = x1
            x2_ = x2
            y1_ = y1
            y2_ = y2

    return x1_, x2_, y1_, y2_
