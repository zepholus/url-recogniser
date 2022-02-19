# adds image processing capabilities
from PIL import Image, ImageEnhance
import pytesseract
import cv2
import requests
import numpy as np
import imutils
import re
import webbrowser

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# noise removal
def remove_noise(image):
    return cv2.medianBlur(image, 5)

def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

# thresholding
def thresholding(image):
    """
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    """

    (thresh, blackAndWhiteImage) = cv2.threshold(image, 130, 255, cv2.THRESH_BINARY)

    return blackAndWhiteImage

# dilation
def dilate(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)

# erosion
def erode(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.erode(image, kernel, iterations=1)

# opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

# canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)

# skew correction
def deskew(image):
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
    return rotated


# template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)

url_dict = {}


def image_ocr(img):

    gray = get_grayscale(img)
    noise = remove_noise(gray)
    darker_gray = adjust_gamma(noise, 0.6)
    thresh = thresholding(darker_gray)
    """
    cv2.imshow('image', thresh)
    cv2.waitKey(0)
    """


    result = pytesseract.image_to_string(thresh)
    print(result)
    url_regex = "(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})"
    regex = re.compile(url_regex, re.DOTALL)
    urls = re.findall(regex, result)
    for url in urls:
        print(url[0])
        webbrowser.open(url[0])
        return True
        """
        if url[0] in url_dict:
            url_dict[url[0]] += 1
        else:
            url_dict[url[0]] = 1
        """
    return False




def main():

    # Replace the below URL with your own. Make sure to add "/shot.jpg" at last.
    url = "http://192.168.0.102:8080/shot.jpg" #Send from your phone using IP WEBCAM app

    # While loop to continuously fetching data from the Url
    while True:
        img_resp = requests.get(url)
        img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
        img = cv2.imdecode(img_arr, -1)

        if image_ocr(img):
            break

        # Press Esc key to exit
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()

    """
    image = cv2.imread('image_test/url.jpg')
    image_ocr(image)
    """



if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
