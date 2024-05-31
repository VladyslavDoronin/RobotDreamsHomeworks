import cv2
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = [15, 10]

#Denoising is the operation of removing (reducing) noise from a signal.

img = cv2.imread('data/kodim07.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Тут просто накладываем шум на исходную картинку в рандомном порядке
noisy = img/255 + 0.1*np.random.randn(*img.shape)
noisy[noisy < 0] = 0
noisy[noisy > 1] = 1
noisy = (255*noisy).astype(np.uint8)

plt.subplot(121), plt.imshow(img), plt.title('Original')
plt.subplot(122), plt.imshow(noisy), plt.title('Noisy')
plt.show()

# Пробуем убрать шум с помощью фильтра Гаусса
# Apply Gaussian low-pass filter to reduce noise
out = cv2.GaussianBlur(noisy, ksize=(5,5), sigmaX=5)
plt.subplot(121), plt.imshow(noisy)
plt.subplot(122), plt.imshow(out)
plt.show()

plt.subplot(121), plt.imshow(noisy[0:300, 0:300, :])
plt.subplot(122), plt.imshow(out[0:300, 0:300, :])
plt.show()

# # Compare with bilateral filter
bilat = cv2.bilateralFilter(noisy, d=9, sigmaColor=75, sigmaSpace=75)
plt.subplot(121), plt.imshow(out)
plt.subplot(122), plt.imshow(bilat)
plt.show()
plt.subplot(121), plt.imshow(out[200:600, 300:600, :])
plt.subplot(122), plt.imshow(bilat[200:600, 300:600, :])
plt.show()