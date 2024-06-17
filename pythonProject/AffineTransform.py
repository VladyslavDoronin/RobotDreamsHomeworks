import cv2
import math
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = [15, 10]
img = cv2.imread('data/image2_res.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
rows, cols, _ = img.shape

plt.imshow(img)
plt.show()

zoom = 1.75
map_x = np.zeros((rows, cols), dtype=np.float32)
map_y = np.zeros((rows, cols), dtype=np.float32)

for r in range(rows):
    for c in range(cols):
        map_x[r, c] = c*1/zoom
        map_y[r, c] = r*1/zoom


print('Map x')
print(map_x[500:505, 500:505]) # Выбрали область по х от и до
print('')
print('Map y')
print(map_y[500:505, 500:505]) # Выбрали область по у от и до

dst = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR) # Из-за того что при c*1/zoom будет не целые числа, и на координатной сетке х,у такого не существует, то делаеся ремап, который по сути округляет наши значения
plt.subplot(121), plt.imshow(img)
plt.subplot(122), plt.imshow(dst)
plt.show()


# Any affine transform can be written in form of matrix multiplication. Given the transform matrix, OpenCV offers a direct way to apply the transform.

Z = np.array([[zoom, 0, 0], [0, zoom, 0]]) # Матрица аффиной трансформации. По сути поворот картинки с наклоном - аффинная трансформация
out = cv2.warpAffine(img, Z, (cols, rows)) # Тут получаем тоже самое что и делали выше с зумом и ремапом. Просто есть такая реализация в опенсиви

plt.subplot(121), plt.imshow(img)
plt.subplot(122), plt.imshow(out)
plt.show()
out = cv2.warpAffine(img, Z, (int(zoom*cols), int(zoom*rows))) # зум обратно, возвращаем наши размеры

plt.subplot(121), plt.imshow(img)
plt.subplot(122), plt.imshow(out)
plt.show()


# Image rotation is also an affine transform.

theta = np.deg2rad(5)
R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0]]) #  Ротация, поворот на угол тета. Тут ротация происходит не от центра, а от координат 0,0(левый нижнийугол)

out = cv2.warpAffine(img, R, (cols, rows))

plt.subplot(121), plt.imshow(img)
plt.subplot(122), plt.imshow(out)
plt.show()

# Для решения проблемы находим центр картинки и его будем считать 0,0
# To account for the rotation not being center, we can build the rotation matrix with a built-in OpenCV function.
center = (cols//2, rows//2)
rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=45, scale=1)

print(rotate_matrix, rows, cols, rows//2, cols//2)

out = cv2.warpAffine(img, rotate_matrix, (cols, rows))

plt.subplot(121), plt.imshow(img)
plt.subplot(122), plt.imshow(out)
plt.show()
