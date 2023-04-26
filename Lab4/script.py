import numpy as np
import cv2 as cv
import imutils

RESOURCE_PATH = 'Lab4/resources/'
OUTPUT_PATH = 'Lab4/results/'
STEP_SIZE = 180


img = cv.imread(f'{RESOURCE_PATH}input.png')
(partW, partH) = (STEP_SIZE, STEP_SIZE)

img = img[90:810, 170:1610]
imgProba = img.copy()

rows = open(f'{RESOURCE_PATH}synset_words.txt').read().strip().split("\n")
classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]
net = cv.dnn.readNetFromCaffe(
    f'{RESOURCE_PATH}bvlc_googlenet.prototxt', f'{RESOURCE_PATH}bvlc_googlenet.caffemodel')

i = 0
stepSizeX = STEP_SIZE
stepSizeY = STEP_SIZE
scale = 2

while img.shape[0] >= partW and img.shape[1] >= partH:
    for y in range(0, img.shape[0], stepSizeY):
        for x in range(0, img.shape[1], stepSizeX):
            partImg = img[y:y+partH, x:x+partW]

            clone = partImg.copy()
            blob = cv.dnn.blobFromImage(
                partImg, 1, (224, 224), (104, 117, 123))

            net.setInput(blob)
            preds = net.forward()

            idxT = (np.argsort(preds[0])[::-1][:5])[0]
            text = ''
            if 'dog' in classes[idxT]:
                text = 'DOG'
            if 'cat' in classes[idxT]:
                text = 'CAT'

            if preds[0][idxT] > 0.9 and text == 'CAT':
                cv.rectangle(imgProba, (x*scale**i, y*scale**i),
                             (x*scale**i + partW*scale**i, y*scale**i + partH*scale**i), (0, 0, 255), 2)
                cv.putText(imgProba, text, (x * scale**i + 5, y * scale**i + 25),
                           cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if preds[0][idxT] > 0.7 and text == 'DOG':
                cv.rectangle(imgProba, (x * scale ** i, y * scale ** i),
                             (x * scale ** i + partW * scale ** i, y*scale**i + partH * scale ** i), (0, 255, 255), 2)
                cv.putText(imgProba, text, (x * scale**i + 5, y * scale**i + 25),
                           cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    w = int(img.shape[1] / scale)
    img = imutils.resize(img, width=w)
    i = i + 1

cv.imshow('Output', imgProba)
cv.imwrite(f'{OUTPUT_PATH}output.jpg', imgProba)
cv.waitKey()
cv.destroyAllWindows()
