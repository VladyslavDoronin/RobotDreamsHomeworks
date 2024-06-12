import cv2
import math
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = [15, 10]
img = cv2.imread('data/checkerboard_clean.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(gray, cmap='gray')
plt.show()

# Harris Corner Detector
# Harris corner detector evaluates each image pixel for the so called "cornerness", i.e., it measures how much the neighbourhood of each pixel changes in different directions. Pixels, whose neighborhood changes both vertically and horizontally are considered corners (or corner candidates).

src = gray.astype(np.float32)
dst = cv2.cornerHarris(src, blockSize=2, ksize=11, k=0.04)
plt.imshow(dst, cmap='gray')
plt.show()

# Remember the cornerness equation: for edge pixel one of the eigenvalues (of the Harris matrix M) is very large and the other very small.This yields negative (with relatively large absolute value) cornerness index.
# Let's us now print the detected corners on the original image.

rows, cols = dst.shape
th = 0.9 * np.max(dst)  # 90% максимального значения найденных углов
result = np.copy(img)

for r in range(rows):
    for c in range(cols):
        if dst[r, c] > th:
            # Для того чтоб видеть детекцию углов, мы на оригинальную картинку накладываем мелкий красный круг
            # Ставиться этот кружок только тогда когда по формуле хариса трешхолд будет больше 90%. Типо определяем угол на 100%
            result = cv2.circle(result, (c, r), 5, (255, 0, 0), -1)

plt.imshow(result,cmap='gray')
plt.show()

img = cv2.imread('data/checkerboard.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

src = gray.astype(np.float32)
dst = cv2.cornerHarris(src, blockSize=2, ksize=3, k=0.04)
plt.subplot(121), plt.imshow(img)
plt.subplot(122), plt.imshow(dst)
plt.show()
rows, cols = dst.shape
th = 0.2 * np.max(dst)
result = np.copy(img)

for r in range(rows):
    for c in range(cols):
        if dst[r, c] > th:
            result = cv2.circle(result, (c, r), 1, (255, 0, 0), -1)

plt.imshow(result)
plt.show()

# Let's try the corner detector on parking cameras.

img = cv2.imread('data/parking.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

src = gray.astype(np.float32)
dst = cv2.cornerHarris(src, blockSize=2, ksize=3, k=0.04)
plt.subplot(121), plt.imshow(img)
plt.subplot(122), plt.imshow(dst)
rows, cols = dst.shape
th = 0.2 * np.max(dst)  # Keep going down to 0.01 -> outliers
result = np.copy(img)

for r in range(rows):
    for c in range(cols):
        if dst[r, c] > th:
            result = cv2.circle(result, (c, r), 3, (255, 0, 0), -1)

plt.imshow(result)
plt.show()