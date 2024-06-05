import cv2
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = [15, 10]

img = cv2.imread('data/autobahn.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()

# Prepare gray scale image
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # РГБ в серый цвет
gray = gray/255 # Нормализируем наши значения

#Sobel Edge Detector
#Sobel edge detector is a symmetric rgadient filter with weighted averaging. It is one of the most popular and effective gradient filters.

grad_x = cv2.Sobel(gray, ddepth=-1, dx=1, dy=0)  # Получаем градиент по х. dx=1
grad_y = cv2.Sobel(gray, ddepth=-1, dx=0, dy=1)  # Получаем градиент по y. dy=1

print(gray.shape, grad_x.shape, grad_y.shape)  # Видим что этоодинаковые по размерам картинки

# Compute magnitude
sobel = np.sqrt(grad_x**2 + grad_y**2)  # Это формула Собеля, для объединения двух градиентов по горизонтали и вертикали в один единный градиент
sobel = sobel/np.max(sobel)  # Нормализируем наши значения. Диапазон 0...1

plt.imshow(sobel, cmap='gray')
plt.show()