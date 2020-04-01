'''This class takes an image path as input, performs preprocessing, identifies the
grid, crops the grid, corrects perspective, writes all these stages to StagesImages folder and
finally slices the grid into 81 cells and returns the 2D array of 81 cell images'''
import cv2
import os
import numpy as np
from PIL import Image

class ImagePreprocessor:

    '''Initializes the Class'''
    def __init__(self, imagepath):
        self.image = cv2.imread(imagepath, 0)
        self.imagePath = imagepath
        self.originalimage = np.copy(self.image)
        self.extractedgrid = None

    '''This function blurs the image, applies thresholding, inverts it and dilates the image'''
    def preprocess_image(self):

        gray = self.image

        #Applying Gaussian Blur to smooth out the noise
        gray = cv2.GaussianBlur(gray, (11, 11), 0)
        try:
            os.remove("StagesImages/1.png")
        except:
            pass
        cv2.imwrite("StagesImages/1.png", gray)

        # Applying thresholding using adaptive Gaussian|Mean thresholding
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C | cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 2)
        try:
            os.remove("StagesImages/2.png")
        except:
            pass
        cv2.imwrite("StagesImages/2.png", gray)

        #Inverting the image
        gray = cv2.bitwise_not(gray)
        try:
            os.remove("StagesImages/3.png")
        except:
            pass
        cv2.imwrite("StagesImages/3.png", gray)

        #Dilating the image to fill up the "cracks" in lines
        kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
        gray = cv2.dilate(gray, kernel)
        self.image = gray
        try:
            os.remove("StagesImages/4.png")
        except:
            pass
        cv2.imwrite("StagesImages/4.png", gray)

    def image_smoothening(self):
        img = self.image
        ret1, th1 = cv2.threshold(img, cv2.THRESH_BINARY, 255, cv2.THRESH_BINARY)
        ret2, th2 = cv2.threshold(th1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        blur = cv2.GaussianBlur(th2, (5, 5), 0)
        ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return th3

    def set_image_dpi(self):
        im = Image.open(self.imagePath)
        length_x, width_y = im.size
        factor = min(1, float(1024.0 / length_x))
        size = int(factor * length_x), int(factor * width_y)
        im_resized = im.resize(size, Image.ANTIALIAS)
        im_resized.save('StagesImages/Improved_dpi.png', dpi=(300, 300))
        # try:
        #     os.remove("StagesImages/5.png")
        # except:
        #     pass
        # cv2.imwrite("StagesImages/5.png", im_resized)
        # return filename

    #skew correction
    def deskew(self):
        image = self.image
        coords = np.column_stack(np.where(image > 0))
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        try:
            os.remove("StagesImages/rotated.png")
        except:
            pass
        cv2.imwrite("StagesImages/rotated.png", rotated)
        return rotated

    def remove_noise_and_smooth(self):
        img = cv2.imread(self.imagePath, 0)
        filtered = cv2.adaptiveThreshold(img.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 41)
        kernel = np.ones((1, 1), np.uint8)
        opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        img = self.image_smoothening()
        or_image = cv2.bitwise_or(img, closing)
        try:
            os.remove("StagesImages/final.png")
        except:
            pass
        cv2.imwrite("StagesImages/final.png", or_image)
        return or_image

        