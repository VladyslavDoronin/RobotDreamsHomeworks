import cv2
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = [15, 10]

img = cv2.imread('data/autobahn.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()

# Set up kernels. Фильтр 2х2
kernel_hor = np.array([[1, -1]])
kernel_ver = np.array([[1], [-1]])

# Prepare gray scale image
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
gray = gray/255 # Нормализируем наши значения

# Convolve discrete gradient kernels with the luminance channel
grad_hor = cv2.filter2D(gray, ddepth=-1, kernel=kernel_hor)  # Градиент по горизонтали. Чем-то похоже на выдавливание на кальке. Но тут накладывается 2д фильтр kernel_hor
grad_ver = cv2.filter2D(gray, ddepth=-1, kernel=kernel_ver)  # Градиент по вертикали

plt.subplot(121), plt.imshow(grad_hor, cmap='gray'), plt.title('Horizontal gradient')
plt.subplot(122), plt.imshow(grad_ver, cmap='gray'), plt.title('Vertical gradient')
plt.show()

#The problem with simple discrete gradients is that they are very sensitive to noise.
plt.subplot(121), plt.imshow(grad_hor[400:, 200:300], cmap='gray'), plt.title('Horizontal gradient')  # Вроде на картинке все нормально выглядит, но если приблизить то видно сильные шумы. В этом и проблема этого способа
plt.subplot(122), plt.imshow(grad_ver[400:, 200:300], cmap='gray'), plt.title('Vertical gradient')  # Для решения этой проблемы дальше привожу способ с средним значения фильтра
plt.show()

#Averaging Filters
#Gradient filters can be convolved with averaging filters to get a less noisy gradient map.

# Set up gradient kernels
Gh = np.array([[1, -1]])
Gv = np.array([[1], [-1]])

# Set up averaging kernels
Ah = 0.5 * np.array([[1, 1]])
Av = 0.5 * np.array([[1], [1]])

# Build separable averaging gradient kernels
from scipy.signal import convolve2d
Hh = convolve2d(Gh, Av)
Hv = convolve2d(Gv, Ah)

print('Horizontal filter \n', Hh)
print(' ')
print('Vertical filter \n', Hv)

grad_avg_hor = cv2.filter2D(gray, ddepth=-1, kernel=Hh)
grad_avg_ver = cv2.filter2D(gray, ddepth=-1, kernel=Hv)

plt.subplot(121), plt.imshow(grad_avg_hor, cmap='gray'), plt.title('Horizontal gradient')
plt.subplot(122), plt.imshow(grad_avg_ver, cmap='gray'), plt.title('Vertical gradient')
plt.show()

block_hor = grad_hor[400:, 200:300]
block_avg_hor = grad_avg_hor[400:, 200:300]

vmin = max(np.min(block_hor), np.min(block_avg_hor))
vmax = min(np.max(block_hor), np.max(block_avg_hor))

plt.subplot(121), plt.imshow(block_hor, cmap='gray', vmin=vmin, vmax=vmax), plt.title('Hv')  # Показываем как уменьшилось влияние шума. Обе приблеженные картинки это градиент по горизонтали
plt.subplot(122), plt.imshow(block_avg_hor, cmap='gray', vmin=vmin, vmax=vmax), plt.title('Hv avg')
plt.show()

print(np.min(block_hor), np.max(block_hor))
print(np.min(block_avg_hor), np.max(block_avg_hor))

# Тоже самое делаем для вертикального градиента
block_ver = grad_ver[400:, 200:300]
block_avg_ver = grad_avg_ver[400:, 200:300]

vmin = max(np.min(block_ver), np.min(block_avg_ver))
vmax = min(np.max(block_ver), np.max(block_avg_ver))

plt.subplot(121), plt.imshow(block_ver, cmap='gray', vmin=vmin, vmax=vmax), plt.title('Gv')
plt.subplot(122), plt.imshow(block_avg_ver, cmap='gray', vmin=vmin, vmax=vmax), plt.title('Hv')
plt.show()

print(np.min(block_ver), np.max(block_ver))
print(np.min(block_avg_ver), np.max(block_avg_ver))