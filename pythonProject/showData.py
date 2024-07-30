import cv2
from matplotlib import pyplot as plt

plt.rcParams['figure.figsize'] = [15, 10]

img = cv2.cvtColor(cv2.imread('data/DronTech/train/473747547_png.rf.024e4bc84ce3ed476da0c7b07eb0b265.jpg'), cv2.COLOR_BGR2RGB)
mask = cv2.imread('data/Masks/mask_train/473747547_png.rf.024e4bc84ce3ed476da0c7b07eb0b265.jpg')
plt.subplot(121), plt.imshow(img)
plt.subplot(122), plt.imshow(mask/255*50)
plt.show()