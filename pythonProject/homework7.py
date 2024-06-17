import cv2
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = [15, 10]

img = cv2.imread('data/document.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)/255
rows, cols = gray.shape
print(rows, cols)

cornerness = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
rows, cols = cornerness.shape
print(rows, cols)

cornerness[cornerness < 0] = 0

cornerness = np.log(cornerness + 1e-6)

th_top_left, th_top_right = -1e6, -1e6
th_bottom_left, th_bottom_right = -1e6, -1e6

opt_top_left, opt_top_right = None, None
opt_bottom_left, opt_bottom_right = None, None

quad_size = 7
for r in range(quad_size, rows - quad_size):
    for c in range(quad_size, cols - quad_size):
        if cornerness[r, c] < -7:
            continue
        block = 255 * gray[r - quad_size:r + quad_size + 1, c - quad_size:c + quad_size + 1]

        quad_top_left = block[0:quad_size, 0:quad_size]
        quad_top_right = block[0:quad_size, quad_size+1:]
        quad_bottom_left = block[quad_size+1:, 0:quad_size]
        quad_bottom_right = block[quad_size+1:, quad_size+1:]

        # Top-left corner
        descriptor = (np.mean(quad_bottom_right) -
                      np.mean(quad_top_left) - np.mean(quad_top_right) - np.mean(quad_bottom_left))
        # Let's detect the best descriptor
        if descriptor > th_top_left:
            th_top_left = descriptor
            opt_top_left = (c, r)

        # Top-right corner
        descriptor = (np.mean(quad_bottom_left) -
                      np.mean(quad_top_left) - np.mean(quad_top_right) - np.mean(quad_bottom_right))
        if descriptor > th_top_right:
            th_top_right = descriptor
            opt_top_right = (c, r)

        # Bottom-left corner
        descriptor = (np.mean(quad_top_right) -
                      np.mean(quad_top_left) - np.mean(quad_bottom_left) - np.mean(quad_bottom_right))
        if descriptor > th_bottom_left:
            th_bottom_left = descriptor
            opt_bottom_left = (c, r)

        # Bottom-right corner
        descriptor = (np.mean(quad_top_left) -
                      np.mean(quad_top_right) - np.mean(quad_bottom_left) - np.mean(quad_bottom_right))
        if descriptor > th_bottom_right:
            th_bottom_right = descriptor
            opt_bottom_right = (c, r)

out = np.copy(img)
out = cv2.circle(out, opt_top_left, 1, (255, 0, 0), -1)
out = cv2.circle(out, opt_top_right, 1, (255, 0, 0), -1)
out = cv2.circle(out, opt_bottom_left, 1, (255, 0, 0), -1)
out = cv2.circle(out, opt_bottom_right, 1, (255, 0, 0), -1)


plt.subplot(121), plt.imshow(img)
plt.subplot(122), plt.imshow(out)
plt.show()

# Document Rectification
# Let's now try to rectify the document. The goal is to bring the four document corners to the image corners. For instance, we want the top-left document corner to become (0, 0), i.e., the top-left corner of the image itself. In that way, we will fill the complete image with document information and we will throw away parts of the images that correspond to background (which are of no use to us).

# Define the matrix of source points corresponding to the 4 document corners.
# The matrix shall have shape (4, 2), i.e., 4 corners x 2 coordinates
# Note: You will need to explicitly use float32 data type
src = np.array([opt_top_left, opt_top_right, opt_bottom_left, opt_bottom_right], dtype=np.float32)
print(src)

# Define the matrix of target (destination) points corresponding to the 4 image corners.
# The matrix shall have shape (4, 2), i.e., 4 corners x 2 coordinates
# Note: You will need to explicitly use float32 data type
# Note2: The order of points in src and dst must be the same
dst = np.array([tuple([0, 0]), tuple([cols, 0]), tuple([0, rows]), tuple([cols, rows])], dtype=np.float32)

# Let's first start with the affine transform for document rectification. The affine transform can be analytically calculated using 3 point pairs. Therefore, let's select the first 3 points and calculate the correspnding transfrom. We will then use the transform to rectify the document.

# Compute the affine transform matrix (you'll have to use getAffineTransform function from OpenCV here)
# Use the first 3 points from your src and dst matrix
M =cv2.getAffineTransform(src[0:3], dst[0:3])

# Build the rectified image using the computed matrix (you'll have to use warpAffine function from OpenCV here)
warped = cv2.warpAffine(img, M, (cols,rows))

# Let's plot the results
plt.subplot(121), plt.imshow(img)
plt.subplot(122), plt.imshow(warped)
plt.show()

# Well, this is not bad by certainly not what we were aiming for. Let's try the last 3 points instead.

# Compute the affine transform matrix (use getAffineTransform)
# Use the last 3 points from your src and dst matrix
M = cv2.getAffineTransform(src[1:4], dst[1:4])

# Build the rectified image using the computed matrix (use warpAffine)
rectified = cv2.warpAffine(img, M, (cols,rows))

# Let's plot the results
plt.subplot(121), plt.imshow(img)
plt.subplot(122), plt.imshow(rectified)
plt.show()

# The result looks different but not better. This approach doesn't seem to be helping then. Let's use all 4 points and let OpenCV estimate (remember that 4 points are too many for an analytical solution) the best fitting affine transform for us. It'll internally apply optimization approaches as well as RANSAC.

# Estimate the optimal affine transform matrix (you'll have to use estimateAffine2D function from OpenCV here)
# estimateAffine2D it returns the best fitting affine matrix as well as the vector of inliers (1 -> inlier,
# 0 -> outlier).
M_estimateAffine2D, inliers = cv2.estimateAffine2D(src, dst)
print(inliers)
print(M_estimateAffine2D)

# Build the rectified image using the computed matrix (use warpAffine)
rectified_estimateAffine2D = cv2.warpAffine(img, M_estimateAffine2D, (cols,rows))

# Let's plot the results
plt.subplot(121), plt.imshow(img)
plt.subplot(122), plt.imshow(rectified_estimateAffine2D)
plt.show()

# There is not much of an improvement either. Let's try homography instead of affine transform. Remember that for computing the homography analytically we need to use 4 pairs of points.

# Compute the homography matrix (you'll have to use getPerspectiveTransform function from OpenCV here)
M_getPerspectiveTransform = cv2.getPerspectiveTransform(src, dst)

# Build the rectified image using the computed matrix (you'll have to use warpPerspective function from OpenCV)
rectified_warpPerspective = cv2.warpPerspective(img, M_getPerspectiveTransform, (cols,rows))

# Let's plot the results
plt.subplot(121), plt.imshow(img)
plt.subplot(122), plt.imshow(rectified_warpPerspective)
plt.show()

# Questions
# The affine transform does not seem to be working well in this case. Why?
# Как сказал Ян, аффинное преобразование сохраняет паралелизм линий.
# А в данной картинке видно что документ наклонен и если провести линии вдоль документа, то они когда-то пересекутся, что быть не может(мы это знаем).
# Соответственно, при применении афииной трансформации, которая берет лишь 3 точки, сохраняется паралелизм лишь 3 линий, а 4 точка(реальная которую нужно взять) уходит за пределы и документ обрезается.
# То есть, афинное преобразование не берет в счет наклон, другими словами гомографию или перспективу


# What can you tell me about the values you have obtained for the inliers vector? What does it mean?
# Мы получили 3 inliers вектора и 1 outlier. Это показывает нам что для афинного преобразования подходят лишь 3 наши точки(те которые указаны как 1 в массиве) и для 1й придется сделать гомографию


# How does the result from homography look? Does it work well enough?
# Результат конечно на много лучше чем при простой афииной трансформации. Но видно что пиксели плывут немного, бьется качество.
# А так свою задачу гомограффия выполнила, она сопоставила все 4 нужных нам угла




