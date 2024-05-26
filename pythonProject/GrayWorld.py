import cv2
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = [15, 5]

# Load your image
img = cv2.imread('data/9.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# plt.imshow(img1)
# plt.show()

# Разделяем на каждый цветовой канал
red, green, blue = cv2.split(img)
# red = img1[:, :, 0]
# green = img1[:, :, 1]
# blue = img1[:, :, 2]

# Compute the mean values for all three colour channels (red, green, blue)
mean_r = np.mean(red)
mean_g = np.mean(green)
mean_b = np.mean(blue)

print(f"mean_r: {mean_r}")
print(f"mean_g: {mean_g}")
print(f"mean_b: {mean_b}")

brightestСolorСhannel = max(mean_r, mean_g, mean_b)
# Hint: You can fix the coefficient of the brightest colour channel to 1.
if brightestСolorСhannel == mean_r:
    kr = 1
    # Следовательно mean_r = kg*mean_g = kb*mean_b
    kg = (kr * mean_r) / mean_g
    kb = (kr * mean_r) / mean_b

elif brightestСolorСhannel == mean_g:
    kg = 1
    kr = (kg * mean_g) / mean_r
    kb = (kg * mean_g) / mean_b
else:
    kb = 1
    kr = (kb * mean_b) / mean_r
    kg = (kb * mean_b) / mean_g

print(f"kr: {kr}")
print(f"kg: {kg}")
print(f"kb: {kb}")

print(f"mean_r*kr: {mean_r*kr}")
print(f"mean_g*kg: {mean_g*kg}")
print(f"mean_b*kb: {mean_b*kb}")


balanced = np.zeros_like(img, dtype=np.float32)
balanced[..., 0] = img[..., 0] * kr
balanced[..., 1] = img[..., 1] * kg
balanced[..., 2] = img[..., 2] * kb

balanced = balanced/255
balanced[balanced > 1] = 1
plt.subplot(121), plt.imshow(img), plt.title('Origin')
plt.subplot(122), plt.imshow(balanced), plt.title('Balanced')
plt.show()