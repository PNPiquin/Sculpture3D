import cv2
import os


def face_recognition(m_color):
    """Search for a face in a bgr image

    :param m_color: color image
    :return: (int, int, int, int) : (x1, x2, y1, y2) --> face contained between (x1, y1) and (x2, y2)
    """
    m_color_ = m_color.copy()
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
    face_opt = []

    for face in faces:
        x1 = int(((face[0] - offset) / color_width) * depth_width)
        y1 = int(((face[1] - offset) / color_height) * depth_height)

        x2 = int(((face[0] + face[2] + offset) / color_width) * depth_width)
        y2 = int(((face[1] + face[3] + offset) / color_height) * depth_height)

        cv2.rectangle(m_color_, (face[0], face[1]), (face[0] + face[2], face[1] + face[3]), (255, 0, 0), 3)

        norm = pow((x2 - x1), 2) + pow((y2 - y1), 2)
        if norm > norm_0:
            norm_0 = norm
            x1_ = x1
            x2_ = x2
            y1_ = y1
            y2_ = y2
            face_opt = face

    cv2.imwrite(os.path.join('Presentation', 'faces.jpg'), m_color_[...,::-1])

    return x1_, x2_, y1_, y2_, face_opt
