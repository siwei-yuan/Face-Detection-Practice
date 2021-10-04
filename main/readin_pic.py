import cv2 as cv
from datetime import datetime


def face_detect_func(img):
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    face_detect = cv.CascadeClassifier('C:/Users/YSW/Desktop/Face-Detection-Practice/venv/Lib/site-packages/cv2/data'
                                       '/haarcascade_frontalface_default.xml')
    face = face_detect.detectMultiScale(gray_img)
    for x, y, w, h in face:
        cv.rectangle(img, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=1)
    cv.imshow('Capturing', img)


def save_img():
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


# call face_detect function
# read img from video
cap = cv.VideoCapture(0)
# cap = cv.VideoCapture('/path')
save_img()

cv.destroyAllWindows()
cap.release()
