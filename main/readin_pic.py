import cv2 as cv
from datetime import datetime
import os
from PIL import Image
import numpy as np


def face_detect_func(img):
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    face_detect = cv.CascadeClassifier('C:/Users/YSW/Desktop/Face-Detection-Practice/venv/Lib/site-packages/cv2/data'
                                       '/haarcascade_frontalface_default.xml')
    face = face_detect.detectMultiScale(gray_img)
    for x, y, w, h in face:
        cv.rectangle(img, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=1)
    cv.imshow('Capturing', img)


def save_img():
    cap = cv.VideoCapture(0)
    while cap.isOpened():
        ret_flag, v_show = cap.read()
        # cv.imshow('Capturing', v_show)
        face_detect_func(v_show)
        key = cv.waitKey(1) & 0xFF
        if key == ord('s'):
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            cv.imwrite('C:/Users/YSW/Desktop/Face-Detection-Practice/imgs/saved_img' + current_time + '.jpg', v_show)
        elif key == ord('q'):
            break


def get_image_labels(path):
    face_samples = []
    ids = []
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    face_detector = cv.CascadeClassifier('C:/Users/YSW/Desktop/Face-Detection-Practice/venv/Lib/site-packages/cv2/data'
                                       '/haarcascade_frontalface_default.xml')
    for image_path in image_paths:
        PIL_image = Image.open(image_path).convert('L')
        img_numpy = np.array(PIL_image, 'uint8')
        faces = face_detector.detectMultiScale(img_numpy)
        for x,y,w,h in faces:
            ids.append(1)
            face_samples.append(img_numpy[y:y+h, x:x+w])
    return face_samples, ids


def face_recognition_func(recognizor):
    cap = cv.VideoCapture(0)
    while cap.isOpened():
        ret_flag, img = cap.read()
        gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        face_detect = cv.CascadeClassifier(
            'C:/Users/YSW/Desktop/Face-Detection-Practice/venv/Lib/site-packages/cv2/data'
            '/haarcascade_frontalface_default.xml')
        face = face_detect.detectMultiScale(gray_img)
        for x, y, w, h in face:
            cv.rectangle(img, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=1)
            id, doubt = recognizor.predict(gray_img[y:y+h, x:x+w])
            if doubt < 60:
                print('Hello Jerry, Welcome back!')
            else:
                print('Who are you?')
        cv.imshow('Capturing', img)
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break


if __name__ == '__main__':
    # get training data
    faces, ids = get_image_labels('./data/')
    # load recognizor
    recognizor = cv.face.LBPHFaceRecognizer_create()
    # train
    recognizor.train(faces, np.array(ids))

    # test on webcam
    face_recognition_func(recognizor)


    # # call face_detect function
    # # read img from video
    # cap = cv.VideoCapture(0)
    # # cap = cv.VideoCapture('/path')
    # save_img()
    #
    # cv.destroyAllWindows()
    # cap.release()
