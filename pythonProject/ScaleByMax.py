import cv2
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = [15, 5]

# img = cv2.imread('data/1.jpeg')  # тут все максимумы 255 белый. Ничего не произойдет при скейле
# img = cv2.imread('data/10.jpeg')
img = cv2.imread('data/14.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Compute the maximum values for all three colour channels (red, green, blue)
max_r = np.max(img[..., 0])  # R
max_g = np.max(img[..., 1])  # G
max_b = np.max(img[..., 2])  # B

print(f"max_r: {max_r}")
print(f"max_g: {max_g}")
print(f"max_b: {max_b}")

scale_coeff_R = 255.0/max_r
scale_coeff_G = 255.0/max_g
scale_coeff_B = 255.0/max_b
print(f"scale_coeff_R: {scale_coeff_R}")
print(f"scale_coeff_G: {scale_coeff_G}")
print(f"scale_coeff_B: {scale_coeff_B}")

balanced = np.zeros_like(img, dtype=np.float32)
balanced[..., 0] = img[..., 0] * scale_coeff_R
balanced[..., 1] = img[..., 1] * scale_coeff_G
balanced[..., 2] = img[..., 2] * scale_coeff_B

balanced = balanced/255
balanced[balanced > 1] = 1
plt.subplot(121), plt.imshow(img), plt.title('Origin')
plt.subplot(122), plt.imshow(balanced), plt.title('Balanced')
plt.show()
