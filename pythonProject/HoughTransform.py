import cv2
import math
import numpy as np
from matplotlib import pyplot as plt

plt.rcParams['figure.figsize'] = [15, 10]
img = cv2.imread('data/yield.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
edges = cv2.Canny(gray, threshold1=100, threshold2=550)

plt.subplot(121), plt.imshow(img)
plt.subplot(122), plt.imshow(edges, cmap='gray')
plt.show()

lines = cv2.HoughLines(edges, rho=1, theta=2 * np.pi / 180, threshold=50)  # theta - это градусы в радианнах, поэтому конвертируем в градусы. rho это пикселей. threshold - если линия пероесекается меньше 50 пикселей  то она даже не возхвращается как результат. Таким образом детектим линии которые пересекают пиксели 50 или больше раз
hough = np.zeros_like(edges)

for i in range(0, len(lines[0:3])):  # В списке линий в самом начале лежат самые длинные линии. В этом месте мы берем только первые три линии, так как знаем что знак(который на картинке треугольный) наш имеет лишь 3 линии
    rho = lines[i][0][0]
    theta = lines[i][0][1]
    a = math.cos(theta)  # Офсет наклон
    b = math.sin(theta)  # Офсет наклон
    x0 = a * rho
    y0 = b * rho
    pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a))) # 1000 Это магичское число, так как не знаем длину линии и поэтому просто продолжаем эти линии до конца картинки
    pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
    cv2.line(hough, pt1, pt2, 255, 1, cv2.LINE_AA)  # Для рисовании матрицы линии рисуем линию от одной точки к другой

plt.subplot(121), plt.imshow(edges, cmap='gray')
plt.subplot(122), plt.imshow(hough, cmap='gray')
plt.show()