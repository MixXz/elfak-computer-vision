import cv2 as cv
import numpy as np

INPUT_IMG_PATH = 'Lab3/resources/'
OUTPUT_PATH = 'Lab3/results/'
FLANN_INDEX_KDTREE = 1


def trim(frame):
    if not np.sum(frame[0]):
        return trim(frame[1:])
    if not np.sum(frame[-1]):
        return trim(frame[:-2])
    if not np.sum(frame[:, 0]):
        return trim(frame[:, 1:])
    if not np.sum(frame[:, -1]):
        return trim(frame[:, :-2])
    return frame


img1 = cv.imread(f"{INPUT_IMG_PATH}input1.png")
img2 = cv.imread(f"{INPUT_IMG_PATH}input2.png")
img3 = cv.imread(f"{INPUT_IMG_PATH}input3.png")

detector = cv.xfeatures2d.SIFT_create()

kp1, des1 = detector.detectAndCompute(img1, None)
kp2, des2 = detector.detectAndCompute(img2, None)

index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

good = []
for m, n in matches:
    if m.distance < 0.6 * n.distance:
        good.append(m)

src_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
dst_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)

M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
output1 = cv.warpPerspective(
    img2, M, (img1.shape[1] + img2.shape[1], img1.shape[0]))
output1[0:img1.shape[0], 0:img1.shape[1]] = img1
output1 = trim(output1)

kp1, des1 = detector.detectAndCompute(output1, None)
kp2, des2 = detector.detectAndCompute(img3, None)

index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

good = []
for m, n in matches:
    if m.distance < 0.6*n.distance:
        good.append(m)

src_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
dst_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)

M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
output2 = cv.warpPerspective(
    img3, M, (output1.shape[1] + img3.shape[1], output1.shape[0]))

emptyImage = np.zeros(
    [output1.shape[0], output1.shape[1]+img3.shape[1], 3], dtype=np.uint8)
emptyImage[0:output1.shape[0], 0:output1.shape[1]] = output1

emptyImage[0:output1.shape[0], output1.shape[1] -
           10:] = output2[0:, output1.shape[1]-10:]
emptyImage = trim(emptyImage)
cv.imshow("Output", emptyImage)
cv.imwrite(f"{OUTPUT_PATH}output.jpg", emptyImage)

cv.waitKey(0)
