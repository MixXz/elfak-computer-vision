import numpy as np
import cv2 as cv

INPUT_IMG_PATH = 'Lab2/resources/input.png'
OUTPUT_PATH = 'Lab2/results/'


def morphological_reconstruction(marker: np.ndarray, mask: np.ndarray):
    kernel = np.ones(shape=(7, 7), dtype=np.uint8) * 255
    while True:
        expanded = cv.dilate(src=marker, kernel=kernel)
        expanded = cv.bitwise_and(src1=expanded, src2=mask)

        if (marker == expanded).all():
            return expanded
        marker = expanded


img = cv.imread(INPUT_IMG_PATH)
img_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

_, mask_circles = cv.threshold(img_gray, 127, 255, cv.THRESH_BINARY_INV)

mask_filtered = cv.morphologyEx(
    mask_circles, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (25, 25)))

img_hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)
img_sat = img_hsv[:, :, 1]

_, copper = cv.threshold(img_sat, 50, 255, cv.THRESH_BINARY)
copper_open = cv.morphologyEx(copper, cv.MORPH_OPEN,
                              cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)))

coin_mask = morphological_reconstruction(copper_open, mask_filtered)

output = cv.bitwise_and(img, img, mask=coin_mask)
cv.imshow('output', output)
cv.waitKey(0)

cv.imwrite(f'{OUTPUT_PATH}coin_mask.png', coin_mask)
cv.imwrite(f'{OUTPUT_PATH}output_coin.png', output)
