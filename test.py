import numpy as np
import cv2


def detectFaces(image_name):
    img = cv2.imread(image_name)
    face_cascade = cv2.CascadeClassifier(
        './haarcascade_frontalface_default.xml')
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    result = []
    for (x, y, width, height) in faces:
        result.append((x, y, x + width, y + height))
    return result


def drawFaces(image_name):
    faces = detectFaces(image_name)
    if faces:
        img1 = cv2.imread(image_name)
        for (x1, y1, x2, y2) in faces:
            cv2.rectangle(img1, (x1, y1), (x2, y2), (55, 255, 155), 5)
        cv2.imshow('img', img1)
        cv2.waitKey(0)


print(drawFaces('./obama.jpg'))
