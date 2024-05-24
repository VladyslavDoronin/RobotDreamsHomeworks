import cv2
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = [15, 5]
from time import time

img = cv2.imread('data/kodim05.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.show()

# Строим гистограмму. Количество всех пикселей по каждому цвету
start = time()
rows, cols = img.shape
hist = np.zeros(256)
for r in range(rows):
    for c in range(cols):
        hist[img[r,c]] = hist[img[r,c]] + 1
print('Elapsed time:', time() - start)
plt.plot(np.arange(0, 256), hist)
plt.grid(True)
plt.xlabel('Pixel color'), plt.ylabel('Number of pixels')
plt.show()

# На основе данных с гистограммы узнгаем функцию cdf.
cdf = np.zeros(256)
for idx, h in enumerate(hist):
    # Делаем сумму цветов по каждому пикселю начиная от 0 до индекса+1. Таким образом строится функция cdf
    cdf[idx] = np.sum(hist[0:idx+1])
# Нормализуем функцию cdf к от 0 до 1
cdf = cdf/np.sum(hist)
plt.plot(cdf), plt.grid(True)
plt.xlabel('Pixel color'), plt.ylabel('CDF')
plt.show()


# На основе функции cdf вытягиваем наш результат. Грубо говоря избавляемся от неиспользуемых цветов и другие часто используемые цвета выравниваем с не часто использемыми
equalized = np.zeros((rows, cols), dtype=np.uint8)#Заполнили массив по количеству строк и колонок с возможными значениями 0-255
for r in range(rows):
    for c in range(cols):
        # Берем каждый пиксель картинки и в зависимости от цвета этого пикселя(к примеру 123) берем значение из массива функции cdf(123).Так как до этого по функции cdf мы посчитали сумму всех цветов пикеселей так вытягивается график нужных цветов
        # Все что было не белое но приблеженное к белому, то становится более светлее, а все что было приблеженное к черному становется темнее
        equalized[r,c] = 255*cdf[img[r,c]]

plt.subplot(121), plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.subplot(122), plt.imshow(equalized, cmap='gray', vmin=0, vmax=255)
plt.show()

# ------------------------------------------------------------------->
# Выше не ефективный способ, долгое вычисление. В нампай есть быстрый метод получения гистограм, еквалайзор(выравнивание по цветам) гистограм без циклов фор.
start = time()
hist, bins = np.histogram(img.ravel(), bins=256, range=(0,255))
# Разница в скорости на моем пк
# Elapsed time: 0.1294398307800293
# Elapsed time: 0.00633549690246582
print('Elapsed time:', time() - start)
plt.plot(bins[0:-1]+0.5, hist), plt.grid(True)
plt.show()

dst = cv2.equalizeHist(img)
plt.subplot(121), plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.subplot(122), plt.imshow(dst, cmap='gray', vmin=0, vmax=255)
plt.show()

# -------------------------------------------------------------------->
# Проблема евалицации гистограммы
img = cv2.imread('data/tire.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.subplot(121), plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.subplot(122), plt.imshow(cv2.equalizeHist(img), cmap='gray', vmin=0, vmax=255)
plt.show()

# Видно из графика проблему. Черный растет слишком агресивно
hist, bins = np.histogram(img.ravel(), bins=256, range=(0,255))
cdf = np.cumsum(hist/np.sum(hist))
plt.plot(255*cdf), plt.axis('square'), plt.grid(True)
plt.xlabel('Input'), plt.ylabel('Output')
plt.show()

# -------------------------------------------------------------------->
# Для решение проблемы есть специальный метод createCLAHE. Он делит картинку на гриды опредедленного количества и проводит еквализацию гистограммы по каждому отдельному куску что дает более лучший результат
# clipLimit это максимальный агрессивный прирост функции. Не будет агресивно растягивать черные цвета
clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(18, 18))

plt.subplot(131), plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.subplot(132), plt.imshow(cv2.equalizeHist(img), cmap='gray', vmin=0, vmax=255)
plt.subplot(133), plt.imshow(clahe.apply(img), cmap='gray', vmin=0, vmax=255)
plt.show()