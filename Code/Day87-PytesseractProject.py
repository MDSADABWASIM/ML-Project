import cv2
import pytesseract
import numpy as np
from scripts.ImagePreprocessor import ImagePreprocessor


# Path to an image
imagePath = r'/Users/inndata_flutter/Downloads/4.jpg'


new_image = ImagePreprocessor(imagePath)
new_image.preprocess_image()
new_image.set_image_dpi()
new_image.deskew()
new_image.remove_noise_and_smooth()


# l: language, chosen English

# oem(OCR Engine modes):
# 0    Legacy engine only.
# 1    Neural nets LSTM engine only.
# 2    Legacy + LSTM engines.
# 3    Default, based on what is available.

# psm(Page segmentation modes):
# 0    Orientation and script detection (OSD) only.
# 1    Automatic page segmentation with OSD.
# 2    Automatic page segmentation, but no OSD, or OCR. (not implemented)
# 3    Fully automatic page segmentation, but no OSD. (Default)
# 4    Assume a single column of text of variable sizes.
# 5    Assume a single uniform block of vertically aligned text.
# 6    Assume a single uniform block of text.
# 7    Treat the image as a single text line.
# 8    Treat the image as a single word.
# 9    Treat the image as a single word in a circle.
# 10    Treat the image as a single character.
# 11    Sparse text. Find as much text as possible in no particular order.
# 12    Sparse text with OSD.
# 13    Raw line. Treat the image as a single text line, bypassing hacks that are Tesseract-specific.

# Adding custom options
custom_config = r'-l mya --oem 3 --psm 4'


# read
img = cv2.imread('StagesImages/4.png', cv2.IMREAD_GRAYSCALE)

text = pytesseract.image_to_string(img, config=custom_config)
print(text)