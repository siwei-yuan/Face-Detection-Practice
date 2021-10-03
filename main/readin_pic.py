import cv2 as cv


def face_detect_func():
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    face_detect = cv.CascadeClassifier('C:/Users/YSW/Desktop/Face-Detection-Practice/venv/Lib/site-packages/cv2/data'
                                       '/haarcascade_frontalface_alt2.xml')
    face = face_detect.detectMultiScale(gray_img, 1.01, 5, 0, (100, 100), (300, 300))
    for x, y, w, h in face:
        cv.rectangle(img, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=1)
    cv.imshow('result', img)


# read face1.jpg
img = cv.imread('face1.jpg')

# call face_detect function
face_detect_func()
while True:
    if ord('q') == cv.waitKey(0):
        break

cv.destroyAllWindows()