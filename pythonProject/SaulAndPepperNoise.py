import cv2
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = [15, 10]

# Имупльсивный или очень сильный шум, с которым наложение фильтров не всегда нормально работает

img = cv2.imread('data/kodim07.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

noisy = np.zeros_like(img)
rows, cols, _ = img.shape

probability = 0.1
for r in range(rows):
    for c in range(cols):
        if np.random.rand() < probability:
            # 50% chance of getting salt or pepper
            if np.random.rand() < 0.5:
                noisy[r, c, :] = 255  # Накладываем шум белого пикселя
            else:
                noisy[r, c, :] = 0  # Накладываем шум черного пикселя
        else:
            noisy[r, c, :] = img[r, c, :]

plt.imshow(noisy)
plt.show()

# Let's apply Gaussian filtering to reduce the noise.
out = cv2.GaussianBlur(noisy, ksize=(5,5), sigmaX=5)
plt.subplot(121), plt.imshow(noisy)
plt.subplot(122), plt.imshow(out)
plt.subplot(121), plt.imshow(noisy[200:600, 300:600, :])
plt.subplot(122), plt.imshow(out[200:600, 300:600, :])
plt.show()


# Gaussian filters do not handle well impulsive noises. We can try median filters instead.
#Для решения такой проблемы как белый черный шум исполльзуют среднее размытие. До этого все функции были линейные, а эта функция нелинейна
median = cv2.medianBlur(noisy, ksize=5)
plt.subplot(121), plt.imshow(noisy[200:600, 300:600, :])
plt.subplot(122), plt.imshow(median[200:600, 300:600, :])
plt.show()