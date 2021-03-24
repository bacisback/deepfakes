
from PIL import Image

import numpy as np
import cv2

# Download XML from:  https://github.com/opencv/opencv/tree/master/data/haarcascades

class FaceRecog(object):

    def __init__(self, margin=0):

        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        #eye_cascade = cv2.CascadeClassifier('config_files/haarcascade_eye.xml')

        self.margin = margin


    def detect(self, img):

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        return faces


    def __call__(self, img):
        """
        """

        # img = cv2.imread(img) 
        # img = img.numpy()
        img = np.array(img)

        detections = self.detect(img)
        t = FaceRecog.getFaces(img, detections, margin=self.margin)[0]


        return Image.fromarray(t)


    def __repr__(self):
        return self.__class__.__name__+'()'


    @staticmethod
    def cutImage(img, x, y, w, h, margin=0):

        y0 = y - margin if (y - margin) >= 0 else 0
        yf = y + w + margin #if (y + w + margin) < img.shape[0] else img.shape[0]

        x0 = x - margin if (x - margin) >= 0 else 0
        xf = x + h + margin #if (x + h + margin) < img.shape[1] else img.shape[1]

        # print(img.shape)

        return img[y0:yf, x0:xf, :]


    @staticmethod
    def getFaces(img, list_boxes, margin=10, display=False):

        list_imgs = []

        for (x,y,w,h) in list_boxes:

            img_c = FaceRecog.cutImage(img, x, y, w, h, margin=20)

            list_imgs.append( img_c )

            if display:

                cv2.imshow('img', img_c)
                cv2.waitKey(0)
                cv2.destroyAllWindows()


        if len(list_boxes) == 0:

            list_imgs.append(img)


        return list_imgs




if __name__ == '__main__':

    img = cv2.imread('test.jpg')

    fase_r = FaceRecog()
    faces = fase_r.detect(img)
    l = FaceRecog.getFaces(img, faces)
