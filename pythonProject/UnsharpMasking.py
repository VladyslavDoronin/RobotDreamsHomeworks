import cv2
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = [15, 10]

img = cv2.imread('data/kodim01.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# plt.imshow(img)
# plt.show()

# Create a blurred img
# Convolution
kernelLow = np.array([[2, 3, 2],
                      [3, 4, 3],
                      [2, 3, 2]])/24
print(kernelLow)
unsharpConvolutionLow = cv2.filter2D(img, ddepth=-1, kernel=kernelLow, borderType=cv2.BORDER_REPLICATE)
# plt.imshow(unsharpConvolutionLow)
# plt.show()

kernelHigh = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
print(kernelHigh)
unsharpConvolutionHigh = cv2.filter2D(img, ddepth=-1, kernel=kernelHigh, borderType=cv2.BORDER_REPLICATE)
# plt.imshow(unsharpConvolutionHigh)
# plt.show()

# Gaussing
unsharpGaussing = cv2.GaussianBlur(img, ksize=(5,5), sigmaX=5)
# plt.imshow(unsharpGaussing)
# plt.show()

plt.subplot(221), plt.imshow(img), plt.title('Original')
plt.subplot(222), plt.imshow(unsharpConvolutionLow), plt.title('Convolution Low')
plt.subplot(223), plt.imshow(unsharpConvolutionHigh), plt.title('Convolution High')
plt.subplot(224), plt.imshow(unsharpGaussing), plt.title('Gaussing')
plt.show()

# Create the difference image (original − unsharp)
# Note: Remember that you are working with uint8 data types. Any addition or substractions
# might result in overflow or underflow, respectively. You can prevent this by casting the images to float.

# difConvolutionLow = cv2.subtract(img, unsharpConvolutionLow)
# difConvolutionHigh = cv2.subtract(img, unsharpConvolutionHigh)
# difGaussing = cv2.subtract(img, unsharpGaussing)
imgFloat = img.astype(np.float32)
difConvolutionLow = imgFloat - unsharpConvolutionLow.astype(np.float32)
difConvolutionHigh = imgFloat - unsharpConvolutionHigh.astype(np.float32)
difGaussing = imgFloat - unsharpGaussing.astype(np.float32)


# Apply USM to get the resulting image using `sharpened = original + (original − unsharp) × amount`
# Note: Again, take care of underflows/overflows if necessary.
# amount = -5 #  Чем-то похоже на мульташное. При этом размытие не убрало
# amount = -1 #  Никакой реакции визуально, как было так и осталось
# amount = 0 # Бесмысленно, математически просто показывает исходную картинку
# amount = 0.5 # Хороший визуальный результат
# amount = 1 # То что изначально сделали sharpering(high) стало размытым, а вот то что было размытым получило хороший результат
amount = 1.5
# amount = 2 # То что было размыто по Гаусу, чтоло более четким(как sharpering изначальный). То что изначально сделали sharpering(high) стало еще хуже
# amount = 3 # Становится более четким, уже не очень похож на оригинал
# amount = 5 # Чем больше тем больше происходит sharpering
sharpenedConvolutionLow = np.clip((imgFloat + difConvolutionLow * amount), 0, 255).astype(np.uint8)
sharpenedConvolutionHigh = np.clip((imgFloat + difConvolutionHigh * amount), 0, 255).astype(np.uint8)
sharpenedGaussing = np.clip((imgFloat + difGaussing * amount), 0, 255).astype(np.uint8)

plt.subplot(221), plt.imshow(img), plt.title('Original')
plt.subplot(222), plt.imshow(sharpenedConvolutionLow), plt.title('Sharpened from Convolution Low')
plt.subplot(223), plt.imshow(sharpenedConvolutionHigh), plt.title('Sharpened from Convolution High')
plt.subplot(224), plt.imshow(sharpenedGaussing), plt.title('Sharpened from Gaussing')
plt.show()
