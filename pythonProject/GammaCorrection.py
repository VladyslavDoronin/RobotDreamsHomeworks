import cv2
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = [15, 5]

def gamma_correction(img, gamma):
    # rows, cols, channels = img.shape
    # out = np.zeros_like(img)#заполняем масисв такой же как у img все нулями
    # for r in range(rows):
    #     for c in range(cols):
    #         for ch in range(channels):
    #             out[r, c, ch] = img[r, c, ch]**gamma #каждому пикселю даем гамму коррекцию
    # return out
    return img**gamma

img = cv2.imread('data/dark.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255.0 #k=1/255

plt.subplot(131), plt.imshow(img), plt.title('Origin')
plt.subplot(132), plt.imshow(gamma_correction(img, gamma=1.5)), plt.title('gamma = 1.5')
plt.subplot(133), plt.imshow(gamma_correction(img, gamma=1/3)), plt.title('gamma = 1/3')
plt.show()

print(np.mean(img))
print(np.mean(gamma_correction(img, gamma=1.5)))
print(np.mean(gamma_correction(img, gamma=1/3)))

colors = np.arange(0, 1, 1/255)
plt.subplot(131), plt.plot(colors, colors**1), plt.xlabel('input'), plt.ylabel('output'), plt.grid(True)
plt.subplot(132), plt.plot(colors, colors**1.5), plt.xlabel('input'), plt.ylabel('output'), plt.grid(True)
plt.subplot(133), plt.plot(colors, colors**1/3), plt.xlabel('input'), plt.ylabel('output'), plt.grid(True)
plt.show()
