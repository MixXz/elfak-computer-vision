import numpy as np
import cv2 as cv

INPUT_IMG_PATH = 'Lab1/resources/input.png'
OUTPUT_PATH = 'Lab1/results/'

input_image = cv.imread(INPUT_IMG_PATH, 0)

f = np.fft.fft2(input_image)
fshift = np.fft.fftshift(f)
mag_spec = 20 * np.log(np.abs(fshift))

normalized_mag_spec = cv.normalize(
    mag_spec, None, 0, 1, norm_type=cv.NORM_MINMAX)

_, binary_mask = cv.threshold(normalized_mag_spec, 0.8, 1, cv.THRESH_BINARY)

h, w = binary_mask.shape
center = (w // 2, h // 2)
radius = 5

# Maska koja služi da skine signal niske frekvencije koji se pojavljuje u binary_mask.
tmp_mask = np.ones(binary_mask.shape, binary_mask.dtype)
tmp_mask = cv.circle(tmp_mask, center, radius, 0, cv.FILLED)

# And tmp i binnary maske uklanja low freq signal.
binary_mask = cv.bitwise_and(binary_mask, tmp_mask)

binary_mask = 1 - binary_mask
fshift_filtered = fshift * binary_mask

f_filtered = np.fft.ifftshift(fshift_filtered)
img_filtered = np.fft.ifft2(f_filtered).real
output_image = np.uint8(img_filtered)

filtered_mag_spec = 20 * np.log(np.abs(fshift_filtered) + 1)

cv.imwrite(f'{OUTPUT_PATH}fft_mag.png', mag_spec)
cv.imwrite(f'{OUTPUT_PATH}fft_mag_filtered.png', filtered_mag_spec)
cv.imwrite(f'{OUTPUT_PATH}output.png', output_image)

cv.imshow("Output", output_image)
cv.waitKey(0)
