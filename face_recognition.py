import numpy as np
import cv2

# Download XML from:  https://github.com/opencv/opencv/tree/master/data/haarcascades

class FaceRecog:

    def __init__(self):

        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        #eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


    def detect(self, img):

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        return faces


    @staticmethod
    def cutImage(x, y, w, h):

        return img[y:y+w, x:x+h, :]


if __name__ == '__main__':

    img = cv2.imread('test.jpg')

    fase_r = FaceRecog()
    faces = fase_r.detect(img)

    for (x,y,w,h) in faces:

        img_c = FaceRecog.cutImage(x, y, w, h)

        cv2.imshow('img', img_c)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
